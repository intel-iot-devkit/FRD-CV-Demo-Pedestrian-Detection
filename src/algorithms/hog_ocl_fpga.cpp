#include "hog_ocl_fpga.hpp"
#include "AOCLUtils/aocl_utils.h"

#define SCALE_GRAN 256
#define NBINS 9
#define BLOCK_SIZE 2
#define CELL_SIZE 8
#define BLOCK_HIST (NBINS * BLOCK_SIZE * BLOCK_SIZE)
#define HIT_THRESHOLD 0.01

#define CONF_LIMIT 20
#define INTERSECT_THRESHOLD 0.5

using namespace aocl_utils;
using namespace ml;
using namespace ml::altera;

void cleanup() { }

void groupRectangles(std::vector<cv::Rect>& rectList,
        std::vector<double>& weights,
        int groupThreshold,
        double eps) {
    if(groupThreshold <= 0 || rectList.empty()) return;

    CV_Assert(rectList.size() == weights.size());

    std::vector<int> labels;
    int nclasses = partition(rectList, labels, cv::SimilarRects(eps));

    std::vector<cv::Rect_<double> > rrects(nclasses);
    std::vector<int> numInClass(nclasses, 0);
    std::vector<double> foundWeights(nclasses, DBL_MIN);
    std::vector<double> totalFactorsPerClass(nclasses, 1);
    int i, j, nlabels = (int)labels.size();

    for( i = 0; i < nlabels; i++ )
    {
        int cls = labels[i];
        rrects[cls].x += rectList[i].x;
        rrects[cls].y += rectList[i].y;
        rrects[cls].width += rectList[i].width;
        rrects[cls].height += rectList[i].height;
        foundWeights[cls] = cv::max(foundWeights[cls], weights[i]);
        numInClass[cls]++;
    }

    for( i = 0; i < nclasses; i++ )
    {
        // find the average of all ROI in the cluster
        cv::Rect_<double> r = rrects[i];
        double s = 1.0/numInClass[i];
        rrects[i] = cv::Rect_<double>(cv::saturate_cast<double>(r.x*s),
                cv::saturate_cast<double>(r.y*s),
                cv::saturate_cast<double>(r.width*s),
                cv::saturate_cast<double>(r.height*s));
    }

    rectList.clear();
    weights.clear();

    for( i = 0; i < nclasses; i++ )
    {
        cv::Rect r1 = rrects[i];
        int n1 = numInClass[i];
        double w1 = foundWeights[i];
        if( n1 <= groupThreshold )
            continue;
        for( j = 0; j < nclasses; j++ )
        {
            int n2 = numInClass[j];

            if( j == i || n2 <= groupThreshold )
                continue;

            cv::Rect r2 = rrects[j];

            int dx = cv::saturate_cast<int>( r2.width * eps );
            int dy = cv::saturate_cast<int>( r2.height * eps );

            if( r1.x >= r2.x - dx &&
                    r1.y >= r2.y - dy &&
                    r1.x + r1.width <= r2.x + r2.width + dx &&
                    r1.y + r1.height <= r2.y + r2.height + dy &&
                    (n2 > std::max(3, n1) || n1 < 3) )
                break;
        }

        if( j == nclasses )
        {
            rectList.push_back(r1);
            weights.push_back(w1);
        }
    }
}
void AlteraHOGAlgorithm::check_ocl_rc(cl_int stat, const char* op) {
    if(stat == CL_SUCCESS) return;

    throw algorithm_init_error(op,
            "An OpenCL error occurred");
}

void AlteraHOGAlgorithm::check_ocl_rc_run(cl_int stat, const char* op) {
    if(stat == CL_SUCCESS) return;

    throw std::runtime_error(op);
}

AlteraHOGAlgorithm::AlteraHOGAlgorithm(const cv::Size& size) {
    m_res = new BoundingBoxesResult();
    m_results.push_back(m_res);

    // Find a CL platform
    cl_platform_id platform = findPlatform("SDK for OpenCL");
    if(platform == NULL) throw algorithm_init_error(
                "Cannot find OpenCL platform",
                "No platform containing 'FPGA SDK'");

    cl_device_id device;
    cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device,
            NULL);
    if(status == CL_DEVICE_NOT_FOUND)
        throw algorithm_init_error(
                "Cannot find device",
                "No attached OpenCL devices");
    else if(status != CL_SUCCESS)
        throw algorithm_init_error(
                "Cannot find device",
                "Failed to query OpenCL devices");

    ctx = clCreateContext(0, 1, &device, &oclContextCallback, NULL,
            &status);
    check_ocl_rc(status, "Failed to create OpenCL context");

    q0 = clCreateCommandQueue(ctx, device, 0, &status);
    check_ocl_rc(status, "Failed to create OpenCL command queues");
    q1 = clCreateCommandQueue(ctx, device, 0, &status);
    check_ocl_rc(status, "Failed to create OpenCL command queues");
    q2 = clCreateCommandQueue(ctx, device, 0, &status);
    check_ocl_rc(status, "Failed to create OpenCL command queues");
    q3 = clCreateCommandQueue(ctx, device, 0, &status);
    check_ocl_rc(status, "Failed to create OpenCL command queues");
    q4 = clCreateCommandQueue(ctx, device, 0, &status);
    check_ocl_rc(status, "Failed to create OpenCL command queues");

    std::string binfile = getBoardBinaryFile("pedestrian_detect", device);
    size_t len;
    const unsigned char* data = loadBinaryFile(binfile.c_str(), &len);
    if(data == NULL) {
        throw algorithm_init_error("Failed to find OpenCL program",
                "Can't find bitstream - is pedestrian_detect.aocx visible?");
    }

    cl_int bin_status;
    pgm = clCreateProgramWithBinary(ctx, 1, &device, &len, &data,
            &bin_status, &status);
    check_ocl_rc(status, "Failed to load OpenCL program");

    status = clBuildProgram(pgm, 1, &device, "", NULL, NULL);
    check_ocl_rc(status, "Failed to compile OpenCL program");

    // Get kernels
    k_svm = clCreateKernel(pgm, "svm", &status);
    check_ocl_rc(status, "Failed to find SVM kernel");
    k_resize = clCreateKernel(pgm, "resize", &status);
    check_ocl_rc(status, "Failed to find resize kernel");
    k_gradient = clCreateKernel(pgm, "gradient", &status);
    check_ocl_rc(status, "Failed to find gradient kernel");
    k_histogram = clCreateKernel(pgm, "histograms", &status);
    check_ocl_rc(status, "Failed to find histogram kernel");
    k_norm = clCreateKernel(pgm, "normalizeit", &status);
    check_ocl_rc(status, "Failed to find normalization kernel");

    // Configure buffers
    d_originalData = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
            (size.width + 128) * (size.height + 128) * sizeof(unsigned int),
            NULL, &status);
    check_ocl_rc(status, "Failed to allocate buffer");

    // Allocate per-level buffers
    for(int i = 0;i < LEVELS;i++) {
        d_inData[i] = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                2*(size.width+128)*(size.height+128)*sizeof(unsigned int),
                NULL, &status);
        check_ocl_rc(status, "Failed to allocate level buffer");
        d_outData[i] = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                2*(size.width+128)*(size.height+128)*sizeof(unsigned int),
                NULL, &status);
        check_ocl_rc(status, "Failed to allocate level buffer");
        h_results[i] = (int*)alignedMalloc((size.width+64)*(size.height+128)*sizeof(int));
    }

    d_imgBuffer = (char*)alignedMalloc(size.height*size.width*4);
    if(d_imgBuffer == NULL) throw algorithm_init_error(
            "Failed to allocate input buffer",
            "Memory allocation failed");
}

AlteraHOGAlgorithm::~AlteraHOGAlgorithm() {
    delete m_res;
}

Algorithm::Info AlteraHOGAlgorithm::getInfo() {
    return m_info;
}

const std::vector<AlgorithmResult*>& AlteraHOGAlgorithm::analyze(const cv::Mat& mat) {
    double scale = 1;
    double scale0 = pow(mat.rows / 128, 1.0/LEVELS);

    int inSize = mat.rows * mat.cols * mat.elemSize();
    memcpy(d_imgBuffer, mat.data, inSize);
    cl_int status = clEnqueueWriteBuffer(q0,
            d_originalData, CL_TRUE, 0,
            inSize, d_imgBuffer, 0, NULL, NULL);
    check_ocl_rc_run(status, "Failed to copy data to device");

    cl_event resized[LEVELS], gradiented[LEVELS], histogrammed[LEVELS],
        normalized[LEVELS], svmed[LEVELS];
    cv::Size _paddingTL(32, 32);
    cv::Size _paddingBR(32, 32);
    std::vector<cv::Rect> locations;
    std::vector<double> weights;

    for(int level=0;level < LEVELS;level++) {
        cl_int scale_int = cvRound((float)SCALE_GRAN / scale);

        cv::Size sz(
                cvRound(mat.cols*scale_int/SCALE_GRAN),
                cvFloor(mat.rows*scale_int/SCALE_GRAN));
        cv::Size gradsize (sz.width + _paddingTL.width + _paddingBR.width,
                sz.height + _paddingTL.height + _paddingBR.height);

        int blX = (gradsize.width + 7) / 8;
        int blY = (gradsize.height + 7) / 8;

        int delta = (_paddingTL.width + _paddingTL.height *
                (sz.width + _paddingTL.width + _paddingBR.width));
        int padding = _paddingTL.width;

        // enqueue resize kernel
        cl_int mrows = mat.rows, mcols = mat.cols;
        check_ocl_rc_run(clSetKernelArg(k_resize, 0, sizeof(cl_int),
                    &scale_int), "Failed to configure resize kernel");
        check_ocl_rc_run(clSetKernelArg(k_resize, 1, sizeof(cl_mem),
                    &d_originalData), "Failed to configure resize kernel");
        check_ocl_rc_run(clSetKernelArg(k_resize, 2, sizeof(cl_int),
                    &mrows), "Failed to configure resize kernel");
        check_ocl_rc_run(clSetKernelArg(k_resize, 3, sizeof(cl_int),
                    &mcols), "Failed to configure resize kernel");
        check_ocl_rc_run(clEnqueueTask(q0, k_resize, 0, NULL, NULL),
                "Failed to queue resize kernel");

        // enqueue gradient kernel
        check_ocl_rc_run(clSetKernelArg(k_gradient, 0, sizeof(cl_int),
                    &sz.height), "Failed to configure gradient kernel");
        check_ocl_rc_run(clSetKernelArg(k_gradient, 1, sizeof(cl_int),
                    &sz.width), "Failed to configure gradient kernel");
        check_ocl_rc_run(clEnqueueTask(q1, k_gradient, 0, NULL, NULL),
                "Failed to queue gradient kernel");

        // enqueue histograms kernel
        check_ocl_rc_run(clSetKernelArg(k_histogram, 0, sizeof(cl_int),
                    &gradsize.height), "Failed to configure histogram kernel");
        check_ocl_rc_run(clSetKernelArg(k_histogram, 1, sizeof(cl_int),
                    &gradsize.width), "Failed to configure histogram kernel");
        check_ocl_rc_run(clSetKernelArg(k_histogram, 2, sizeof(cl_int),
                    &padding), "Failed to configure histogram kernel");
        check_ocl_rc_run(clEnqueueTask(q2, k_histogram, 0, NULL, NULL),
                "Failed to queue histogram kernel");

        // enqueue normalize kernel
        check_ocl_rc_run(clSetKernelArg(k_norm, 0, sizeof(cl_int),
                    &gradsize.height), "Failed to configure norm kernel");
        check_ocl_rc_run(clSetKernelArg(k_norm, 1, sizeof(cl_int),
                    &gradsize.width), "Failed to configure norm kernel");
        int pixels = (blX*blY+2)*BLOCK_HIST;
        check_ocl_rc_run(clSetKernelArg(k_norm, 2, sizeof(cl_int),
                    &pixels), "Failed to configure norm kernel");
        int pixwrite = (gradsize.height) / CELL_SIZE * ((
                        gradsize.width + CELL_SIZE - 1)
                        / CELL_SIZE)*BLOCK_HIST;
        check_ocl_rc_run(clSetKernelArg(k_norm, 3, sizeof(cl_int),
                    &pixwrite), "Failed to configure norm kernel");
        check_ocl_rc_run(clEnqueueTask(q3, k_norm, 0, NULL, NULL),
                "Failed to queue norm kernel");

        // enqueue svm kernel
        check_ocl_rc_run(clSetKernelArg(k_svm, 0, sizeof(cl_mem),
                    &d_inData[level]), "Failed to configure SVM kernel");
        check_ocl_rc_run(clSetKernelArg(k_svm, 1, sizeof(cl_int), &blX),
                "Failed to configure SVM kernel");
        check_ocl_rc_run(clSetKernelArg(k_svm, 2, sizeof(cl_int), &blY),
                "Failed to configure SVM kernel");
        check_ocl_rc_run(clEnqueueTask(q4, k_svm, 0, NULL, NULL),
                "Failed to launch SVM kernel");

        int outSize = blX * blY * sizeof(float);
        clEnqueueReadBuffer(q4, d_inData[level], CL_FALSE, 0,
                outSize, h_results[level], 0, NULL, NULL);

        // update scale
        if(cvRound(mat.cols / scale) < 64 || cvRound(mat.rows / scale) < 64
                || scale0 <= 1)
            break;
        scale *= scale0;
        scale = SCALE_GRAN / scale;
        scale = cvRound(scale);
        scale = SCALE_GRAN / scale;
    }
    clFinish(q4);

    scale = 1;
    for(int level = 0;level < LEVELS;level++) {
        cv::Size sz(
                cvFloor(mat.cols/scale),
                cvFloor(mat.rows/scale));
        int scale_int = cvRound((float)SCALE_GRAN / scale);
        cv::Size gradsize(
                sz.width + _paddingTL.width + _paddingBR.width,
                sz.height + _paddingTL.height + _paddingBR.height);
        int blX = (gradsize.width + 7) / 8; 
        int blY = (gradsize.height + 7) / 8; 
        int outSize = blX * blY * sizeof(float);
        int where = 0;

        for(int y = -_paddingTL.height, by = 0;by < (blY - 16);by++, y += 8) {
            for (int x = -_paddingTL.width, bx = 0; bx < blX - 8 + 2; bx++, x += 8) {
                float s = h_results[level][where++];
                if (s >= HIT_THRESHOLD) {
                    locations.push_back(cv::Rect(
                                (int)(x * scale),
                                (int)((y + 8) * scale),
                                (int)(64 * scale),
                                (int)(128 * scale)));
                    weights.push_back(s);
                }
            }
        }

        if( cvRound(mat.cols/scale) < 64 ||
                cvRound(mat.rows/scale) < 128 ||
                scale0 <= 1 )
            break;
        scale *= scale0;

        scale = SCALE_GRAN / scale;
        scale = cvRound(scale);
        scale = SCALE_GRAN / scale;
    }

    groupRectangles(locations, weights, 1, 0.2);

    m_res->boxes.clear();

    // update all trackers
    for(std::list<TrackingInfo>::iterator i = m_track.begin();
            i != m_track.end();) {
        cv::Rect2d r;
        if(!i->tracker->update(mat, r)) {
            i = m_track.erase(i);
            continue;
        }
        i->last_pos = r;
        i->confirm_frames++;
        i++;
    }

    // isolate bounds that already have detected people
    for(auto i : m_track) {
        for(int j = 0;j < locations.size();j++) {
            cv::Rect r_t = i.last_pos;
            cv::Rect r_d = locations[j];
            if(r_t.contains(r_d.tl()) && r_t.contains(r_d.br())) {
                // readjust tracking rectangle
                i.confirm_frames = 0;
                i.last_pos = r_t;
                i.tracker->init(mat, r_t);
                locations.erase(locations.begin()+j);
                j--; // revisit the same index next time around
                continue;
            }
            cv::Rect isect = r_t & r_d;
            if((isect.area() >= INTERSECT_THRESHOLD*r_t.area()) &&
                    (isect.area() >= INTERSECT_THRESHOLD*r_d.area())) {
                // they intersect - update the confirm count
                i.confirm_frames = 0;
                locations.erase(locations.begin()+j);
                j--; // revisit the same index next time around
            }
        }
    }

    // delete old trackers
    m_track.remove_if(
            [](TrackingInfo i) { return i.confirm_frames > CONF_LIMIT; });

    // create new tracking bounds for others
    for(auto r : locations) {
        TrackingInfo inf;
        inf.tracker = cv::Tracker::create("TLD");
        inf.last_pos = r;
        inf.id = rand();
        inf.confirm_frames = 0;
        inf.tracker->init(mat, r);
        m_track.push_back(inf);
    }

    // construct results
    for(auto t : m_track) {
        BoundingBox b;
        b.id = t.id;
        b.tag = 0;
        b.bounds = t.last_pos;
        m_res->boxes.push_back(b);
        printf("%d\n", t.confirm_frames);
    }

    /*
    for(int i = 0;i < locations.size();i++) {
        cv::Rect r = locations[i];

        int j;
        for(j = 0;j < locations.size();j++) {
            if(j != i && (r & locations[j]) == r) break;
        }
        if(j == locations.size()) {
            BoundingBox b;
            b.id = 0;
            b.tag = 0;
            b.bounds = r;
            m_res->boxes.push_back(b);
        }
    }
    */

    return m_results;
}

extern "C" int count() {
    return 1;
}

extern "C" Algorithm* build(int idx, const cv::Size& sz) {
    return new AlteraHOGAlgorithm(sz);
}

extern "C" Algorithm::Info* describe(int idx) {
    return new Algorithm::Info(
        "OpenCL FPGA-based HOG SVM", "hog-ocl-fpga",
        "Altera's HOG SVM classifier running on an FPGA via OpenCL",
        0, false, true);
}

extern "C" void interface_version(int* major, int* minor) {
    *major = IFACE_VERSION_MAJOR;
    *minor = IFACE_VERSION_MINOR;
}
