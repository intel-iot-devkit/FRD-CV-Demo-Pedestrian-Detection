// Copyright (C) 2016-2019 Intel Corporation. All rights reserved. Permission
// is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the
// Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions: The above copyright
// notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of
// California and by the laws of the United States of America.

#include "opencv2/core/core.hpp"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <memory>

#include <boost/program_options.hpp>
#include <boost/format.hpp>

#include "config.h"

#include "media/capture.hpp"
#include "media/sink.hpp"
#include "ui.hpp"
#include "algorithm.hpp"
#include "results.hpp"

#ifdef __linux__
#include <unistd.h>
#endif

// define USE_X11 to replace the OpenCV viewer with X11 viewer 
#ifdef USE_X11
#include <X11/Xlib.h>
#endif

using namespace cv;
using namespace std;
namespace po = boost::program_options;

double getTime() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (((double)t.tv_nsec) / 1.0e9);
}

bool showtext = false;
bool verbose = false;
int level=13;

/** Set up options description and parse command-line options */
po::variables_map read_options(int argc, char **argv) {
    ml::AlgorithmRegistry& algoReg = ml::AlgorithmRegistry::get();

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Print this help message")
        ("text,t", "Enable text display")
        ("no-window,w", "Run in command-line-only mode (non-windowed)")
        ("verbose,v", "Enable verbose output")
        ("infinite,i", "Try to make sure video stream doesn't terminate")
        ("vstream,V", po::value<string>(), 
#ifdef NETWORK_OUTPUT
        "Stream video to given host")
#else
        "Stream video to given host (disabled)")
#endif
        ("mstream,M", po::value<string>(), "Stream metadata to given host")
        ("algorithm,a",
             po::value<vector<string> >()->default_value({"ocv-hog-svm"},
                 "ocv-hog-svm"),
            "Specify video processing algorithm to use")
        ("list-algos", "List all available algorithm modules");

    po::options_description hidden_desc;
    hidden_desc.add_options()
        ("input", po::value<string>(), "")
        ("output", po::value<string>(), "");

    po::options_description parse_desc;
    parse_desc.add(desc);
    parse_desc.add(hidden_desc);

    po::positional_options_description pos_opts;
    pos_opts.add("input", 1);
    pos_opts.add("output", 1);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv)
                .options(parse_desc).positional(pos_opts).run(),
                vm);

        if(vm.count("help") > 0) {
            cout << desc << '\n';
            exit(0);
        }

        // handle list command
        if(vm.count("list-algos") > 0) {
            auto list = algoReg.getList();
            printf("Available algorithms:\n");
            for(auto i : list) {
                printf("   %16s - %s\n", i->shortname.c_str(), i->desc.c_str());
            }
            exit(0);
        }

        // sanity checks
        if(vm.count("input") == 0) {
            cerr << "Error: you must specify an input stream\n";
            exit(1);
        }

        po::notify(vm);
    } catch(po::required_option& e) {
        cerr << "Error: " << e.what() << '\n';
        exit(1);
    } catch(po::error& e) {
        cerr << "Error: " << e.what() << '\n';
        exit(1);
    }
    return vm;
}

/** Configure a fanout sink with all output targets for the given options */
void configure_sink(const po::variables_map &vm, vio::FanoutSink& sink) {
    if(vm.count("output") > 0) {
        vio::FileSink* fsink = new vio::FileSink(
                vm["output"].as<string>(),
                vio::FileSink::MPEG4);
        sink.addSink(fsink);
    }
    if(getenv("DISPLAY") != NULL && !vm.count("no-window")) {
        vio::HighGUISink* wsink = new vio::HighGUISink("pedestrian detect");
        sink.addSink(wsink);
    }
    if(vm.count("vstream") > 0) {
#ifdef NETWORK_OUTPUT
        vio::GStreamerSink* nsink = new vio::GStreamerSink(boost::str(
                boost::format("appsrc ! videoconvert ! x264enc qp-min=18 !"
                "rtph264pay ! udpsink host=%1% port=5501")
                % vm["vstream"].as<string>()), 10);
        sink.addSink(nsink);
#else
        cerr << "Error: This binary was not built with network output support.\n";
#endif
    }
}

int main(int argc, char** argv) {
    ml::AlgorithmRegistry& algoReg = ml::AlgorithmRegistry::get();

    // process command-line options
    po::variables_map vm = read_options(argc, argv);

    verbose = vm.count("verbose") > 0;
    showtext = vm.count("text") > 0;

    // open video capture
    vio::CaptureBackend* vcap = vio::openBackend(
            vm["input"].as<string>(),
            vm.count("infinite") > 0);

    // set up video sink
    vio::FanoutSink sink;
    configure_sink(vm, sink);

    // create the algorithm(s)
    ml::Algorithm* algo; // the main algorithm to use
    algoReg.setSize(vcap->getSize());
    try {
        vector<string> goal = vm["algorithm"].as<vector<string> >();
        if(goal.size() == 1) { // just load the target algorithm
            algo = ml::AlgorithmRegistry::get().load(goal[0]);
            if(algo == NULL) {
                fprintf(stderr, "Error: Cannot load algorithm: %s\n", goal[0].c_str());
                fprintf(stderr, "       Use --list-algos to show available options\n");
                return 1;
            } else if(verbose) {
                ml::Algorithm::Info inf = algo->getInfo();
                printf("Loaded %s\n", inf.name.c_str());
            }
        } else { // load multiple algorithms
            // set up a composite group
            ml::CompositeAlgorithm* group = new ml::CompositeAlgorithm();
            algo = group;
            for(auto a : goal) {
                auto r = ml::AlgorithmRegistry::get().load(goal[0]);
                if(r == NULL) {
                    fprintf(stderr, "Error: Cannot find algorithm: %s\n", a.c_str());
                } else {
                    if(verbose) {
                        ml::Algorithm::Info inf = algo->getInfo();
                        printf("Loaded %s\n", inf.name.c_str());
                    }
                    group->add(r);
                }
            }
        }
    } catch(ml::algorithm_init_error& e) {
        fprintf(stderr, "%s\nError: Failed to initialize algorithm: %s\n", e.what(),
                vm["algorithm"].as<string>().c_str());
        return 1;
    }
    bool isFPGAAlgo = algo->getInfo().fpga;

    // set up UI and register fields
    ui::TUIManager tuiMgr;
    ui::CPULoad *cpuLoad = new ui::CPULoad();
    tuiMgr.registerField("cpu", 'c', cpuLoad);
    tuiMgr.registerField("mode", 'm',
            new ui::StaticField(isFPGAAlgo ? "fpga" : "cpu"));

    ui::ValueField* fps = new ui::ValueField();
    fps->setAlpha(0.9);
    tuiMgr.registerField("fps", 'f', fps);

    ui::StatusLine termStatus("[{mode/4}] {fps/3} FPS | CPU: {cpu}%", &tuiMgr);

    // set up the visual overlay
    ui::Overlay overlay;
    ui::ResultRenderElement *resultDraw = NULL;
    ui::StackedBarElement *stackBar = NULL;
    if(!sink.empty()) {
        overlay.add(new ui::TextUIElement("CPU: {cpu/6}%", &tuiMgr,
                    cv::Point2f(0.0005,0.0005), cv::Scalar(255,255,255),
                    ui::TextUIElement::alignment::NORTH_WEST));
        if(isFPGAAlgo) {
            stackBar = new ui::StackedBarElement(cv::Point2f(0.5, 0.7), 10);
            stackBar->setWidth(50);
            overlay.add(stackBar);
        }

        resultDraw = new ui::ResultRenderElement();
        overlay.add(resultDraw);
    }

    // set up metadata dumper if needed
    mdump::Metadumper* dumper = NULL;
    if(vm.count("mstream") > 0) {
        std::unique_ptr<mdump::TCPTarget> tgt(new mdump::TCPTarget(
                    vm["mstream"].as<string>().c_str(), "5500"));
        dumper = new mdump::Metadumper(std::move(tgt));
    }

    Mat img, algo_img;
    double dtime;

    // Frame-by-frame processing loop.
    double sttime = getTime();
    double ktime;
    long frame = 0;
    while(vcap->getFrame(img))
    {
        if(isFPGAAlgo) cvtColor(img, algo_img, CV_BGR2BGRA);
        else algo_img = img;

        double time = getTime();
        const std::vector<ml::AlgorithmResult*>* res;
        try {
            res = &algo->analyze(algo_img);
        } catch(const std::exception& e) {
            fprintf(stderr, "Error: %s\n", e.what());
            break;
        }
        dtime = getTime() - time;
        fps->addSample(1.0/dtime);

        if(showtext) {
            printf("\r%s", termStatus.render().c_str());
            fflush(stdout);
        }
        if(resultDraw) resultDraw->setResults(*res);
        overlay.render(img);

        if(dumper) dumper->accept(
                *res, 15, frame, isFPGAAlgo,
                cpuLoad->getValue(), 1.0/dtime, (int)(dtime*1000));

        // show or save the video result
        sink << img;
        tuiMgr.update();

        frame++;
    }
    sink.close();
    return 0;
}

