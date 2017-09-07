#include "algorithm.hpp"

// built-in algorithms
#include "algorithms/ocv.hpp"

#include <stdio.h>
#include <string.h>
#include <dlfcn.h>

#include <stdexcept>

using namespace ml;

Algorithm::Info::Info(const Info& src) {
    name = std::string(src.name.c_str());
    shortname = std::string(src.shortname.c_str());
    desc = std::string(src.desc.c_str());
    file = std::string(src.file.c_str());
    index = src.index;
    tracks = src.tracks;
    fpga = src.fpga;
}

Algorithm::Info::Info(const char* name, const char* shortname, const char* desc,
        int index, bool tracks, bool fpga) : name(name), shortname(shortname),
        desc(desc), file(""), index(index), tracks(tracks), fpga(fpga) { }

algorithm_init_error::algorithm_init_error(const std::string& what,
        const std::string& reason) : std::runtime_error(what) {
    char buf[what.length() + 3 + reason.length() + 1];
    snprintf(buf, what.length() + 2 + reason.length() + 1,
            "%s: %s\n", what.c_str(), reason.c_str());
    m_reason = buf;
}

CompositeAlgorithm::CompositeAlgorithm() {
    m_info = new Algorithm::Info("composite",
            "Composite Algorithm",
            "A collection of sub-algorithms operating on the same input data",
            0, false, false);
}

CompositeAlgorithm::~CompositeAlgorithm() {
    delete m_info;
    for(auto a : m_contents) delete a;
}

void CompositeAlgorithm::add(Algorithm* algo) {
    m_info->fpga |= algo->getInfo().fpga;
    m_info->tracks |= algo->getInfo().tracks;
    m_contents.push_back(algo);
}

Algorithm::Info CompositeAlgorithm::getInfo() { return *m_info; }

const std::vector<AlgorithmResult*>& CompositeAlgorithm::analyze(const cv::Mat& mat) {
    m_results.clear();

    for(auto a : m_contents) {
        auto r = a->analyze(mat);
        m_results.insert(m_results.begin(), r.begin(), r.end());
    }
    return m_results;
}

const char* algorithm_init_error::what() const noexcept {
    return m_reason.c_str();
}

static AlgorithmRegistry* registryInstance = NULL;

AlgorithmRegistry::AlgorithmRegistry() {
    Algorithm::Info *ocvInfo = ml::ocv::describe(0);
    ocvInfo->file = "<built in>";
    m_compiled.push_back(std::make_pair(ocvInfo, (void*)&ml::ocv::build));

    fs::path algos("algorithms");
    if(fs::exists(algos) && fs::is_directory(algos))
        m_searchPaths.push_back(algos);
    m_searchPaths.push_back(fs::path("."));

    m_algos[NULL] = std::vector<Algorithm*>();
    rebuildDatabase();
}

AlgorithmRegistry::~AlgorithmRegistry() {
}

AlgorithmRegistry& AlgorithmRegistry::get() {
    if(registryInstance == NULL)
        registryInstance = new AlgorithmRegistry();
    return *registryInstance;
}

const std::vector<Algorithm::Info*> AlgorithmRegistry::getList() const {
    return m_known;
}

void AlgorithmRegistry::setSize(const cv::Size& sz) {
    m_imsize = sz;
}

Algorithm* AlgorithmRegistry::load(const std::string& name) {
    Algorithm::Info* info = NULL;
    for(auto e : m_known) {
        if(e->shortname.compare(name) == 0)
            info = e;
    }

    if(info == NULL) return NULL;
    return load(*info);
}

Algorithm* AlgorithmRegistry::load(const Algorithm::Info& info) {
    // see if it's built in
    void* build_ptr = NULL;
    void* lib = NULL;
    for(auto p : m_compiled) {
        if(p.first->shortname == info.shortname && p.first->file == info.file) {
            build_ptr = p.second;
            break;
        }
    }

    if(build_ptr == NULL) {
        // find the build() method
        std::map<std::string, void*>::iterator itr = m_libraries.find(info.file);
        if(itr == m_libraries.end()) { // load the library if needed
            lib = dlopen(info.file.c_str(), RTLD_NOW | RTLD_LOCAL);
            if(lib == NULL)
                throw algorithm_init_error("Cannot load algorithm",
                        "Failed to load shared library");

            m_libraries[info.file] = lib;
            m_algos[lib] = std::vector<Algorithm*>();
        } else {
            lib = itr->second;
        }

        build_ptr = dlsym(lib, "build");
    }
    if(build_ptr == NULL) {
        fprintf(stderr, "FATAL: Algorithm entry point is NULL.\n"
                "       This should be impossible.\n");
        abort();
    }

    Algorithm* (*build)(int, const cv::Size&) =
        (Algorithm* (*)(int, const cv::Size&))build_ptr;

    // instantiate the object
    Algorithm* algo = build(info.index, m_imsize);
    m_algos[lib].push_back(algo);
    return algo;
}

void AlgorithmRegistry::unload(Algorithm* algo) {
}

void AlgorithmRegistry::search(fs::path dir) {
    if(!fs::exists(dir) || !fs::is_directory(dir))
        throw std::invalid_argument("Not a valid search directory");
    m_searchPaths.push_back(dir);
}

void AlgorithmRegistry::rebuildDatabase() {
    for(auto e : m_known) delete e;
    m_known.clear();

    // populate compiled algos
    for(auto a : m_compiled) m_known.push_back(a.first);

    // search all paths
    for(auto p : m_searchPaths) {
        try {
            for(auto e : fs::directory_iterator(p))
                indexDirectory(e.path());
        } catch(fs::filesystem_error& e) {
            fprintf(stderr, "Failed to read %s: %s. Skipping",
                    p.c_str(), e.what());
            continue;
        }
    }
}

void AlgorithmRegistry::indexDirectory(fs::path pth) {
    // only load shared objects
    if(pth.extension() != ".so") return;

    // try opening it
    void* lib = dlopen(pth.c_str(), RTLD_NOW | RTLD_LOCAL);
    if(lib == NULL) {
        fprintf(stderr, "%s\n", dlerror());
        return;
    }

    // check that the basic module interface is present
    void* count = dlsym(lib, "count");
    void* build = dlsym(lib, "build");
    void* describe = dlsym(lib, "describe");
    void* iface_vsn = dlsym(lib, "interface_version");
    if(count == NULL || build == NULL || describe == NULL || iface_vsn == NULL) {
        dlclose(lib);
        return;
    }

    // make sure the interface versions match up
    int major, minor;
    void (*f_iface_vsn)(int*,int*) = (void (*)(int*,int*))iface_vsn;
    f_iface_vsn(&major, &minor);
    if(major != IFACE_VERSION_MAJOR || minor != IFACE_VERSION_MINOR) {
        fprintf(stderr, "Warning: Library '%s' is using incompatible interface"
                        " version.\n         Are both the library and demo "
                        "application up-to-date?\n",
                        pth.c_str());
        dlclose(lib);
        return;
    }

    // get descriptions and register algorithms
    Algorithm::Info* (*f_describe)(int) = (Algorithm::Info* (*)(int))describe;
    int (*f_count)(void) = (int (*)(void))count;

    int len = f_count();
    for(int i = 0;i < len;i++) {
        Algorithm::Info* inf = f_describe(i);
        inf->file = pth.c_str();
        m_known.push_back(inf);
    }

    // unload it
    dlclose(lib);
}
