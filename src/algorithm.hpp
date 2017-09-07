#ifndef ALGORITHM_HPP
#define ALGORITHM_HPP

#include <vector>
#include <utility>
#include <string>
#include <map>
#include <stdexcept>

#include <boost/filesystem.hpp>

#include "opencv2/core/core.hpp"

#define IFACE_VERSION_MAJOR 0
#define IFACE_VERSION_MINOR 4

namespace ml {

namespace fs = boost::filesystem;

class algorithm_init_error : public std::runtime_error {
public:
    algorithm_init_error(const std::string& what,
            const std::string& reason);

    virtual const char* what() const noexcept;
    
private:
    std::string m_reason;
};

enum ResultType {
    RT_BOUNDING_BOXES, //!< Results are bounding boxes around parts of the frame
    RT_POINTS,         //!< Results are points in image coordinate system
    RT_CLASSIFICATION, //!< Result is a list of classifications
};

//! The basic algorithm result structure
struct AlgorithmResult {
    ResultType type;

    virtual ~AlgorithmResult() {};
};

struct BoundingBox {
    /** \brief The box's ID, if available
     * If the algorithm doesn't track boxes between multiple frames, this is set
     * to zero. Zero is not a valid ID otherwise.
     */
    unsigned int id;

    /** \brief The box tag, if available
     * This is used to attach metadata to bounding boxes. If metadata is not
     * available, it will be set to zero. Zero is not a valid tag otherwise.
     */
    unsigned int tag;

    //! The axis-aligned bounds of this box in image coordinates
    cv::Rect bounds;
};

//! Algorithm result subclass for RT_BOUNDING_BOXES results
struct BoundingBoxesResult : public AlgorithmResult {
    //! The list of boxes detected in the frame
    std::vector<BoundingBox> boxes;
};

struct Classification {
    //! The class's name, if relevant. Empty string if no name is available.
    std::string name;

    //! The class's ID number. Zero signifies no ID number present.
    unsigned int id;

    //! The class's tag, or 0 if unused. Associates it with other result objects
    unsigned int tag;
};

//! Algorithm result subclass for RT_CLASSIFICATION results
struct ClassificationResult : public AlgorithmResult {
    //! The list of classes assigned to the image frame
    std::vector<Classification> classes;
};

//! A computer vision algorithm
class Algorithm {
public:
    struct Info {
        Info(const Info& src);
        Info(const char* name, const char* shortname, const char* desc,
                int index, bool tracks, bool fpga);

        std::string name; //!< The algorithm's long name
        std::string shortname; //!< The algorithm's short name
        std::string desc; //!< The algorithm's description
        std::string file; //!< The file this algorithm was loaded from
        int index; //!< The algorithm's file-specific index

        bool tracks; //!< Whether this algorithm tracks objects between frames
        bool fpga; //!< Whether this algorithm runs on an FPGA
    };

    //! Query the algorithm for its properties
    virtual Info getInfo()=0;

    //! Nondestructively process a frame
    virtual const std::vector<AlgorithmResult*>& analyze(const cv::Mat& mat)=0;

protected:
    std::vector<AlgorithmResult*> m_results;
};

//! Composite algorithm for executing one or more child algorithms
class CompositeAlgorithm : public Algorithm {
public:
    CompositeAlgorithm();
    ~CompositeAlgorithm();

    /**\brief Add an algorithm to the composite collection
     *
     * The given algorithm will be executed with each input frame, and any
     * results produced will be merged into the composite output collection.
     * By calling this method, ownership of the algorithm is passed to the
     * composite algorithm. The passed object will be deleted when the composite
     * is.
     */
    void add(Algorithm* algo);

    Algorithm::Info getInfo();
    const std::vector<AlgorithmResult*>& analyze(const cv::Mat& mat);

private:
    Algorithm::Info *m_info;
    std::vector<Algorithm*> m_contents;
};

//! Registry singleton for all available algorithms. Owns algorithm objects.
class AlgorithmRegistry {
public:
    static AlgorithmRegistry& get();

    //! Set the default image siz
    void setSize(const cv::Size& size);

    //! Get a list of all known algorithms
    const std::vector<Algorithm::Info*> getList() const;

    /**\brief Load an algorithm by name
     *
     * This utility method will find the given algorithm, then load it. If the
     * load fails or if the algorithm isn't found, this function will return
     * NULL.
     *
     * \returns The new algorithm object.
     */
    Algorithm* load(const std::string& name);

    /**\brief Load an algorithm
     *
     * This will do anything needed to load the given algorithm and return a
     * new instance of it. The resulting instance is owned by this registry, and
     * will be automatically deleted when the registry is, or when it is
     * explicitly unloaded
     *
     * \returns The new Algorithm object
     */
    Algorithm* load(const Algorithm::Info& info);

    /**\brief Unload the given algorithm
     *
     * This will unload the given algorithm, potentially closing its library if
     * it was loaded from a file.
     */
    void unload(Algorithm* algo);

    /**\brief Add a directory to the search path
     *
     * The registry will search the given path for algorithm files and add them
     * all to its database.
     */
    void search(fs::path dir);

private:
    AlgorithmRegistry();
    ~AlgorithmRegistry();

    cv::Size m_imsize;

    //! Rebuild database by searching known paths
    void rebuildDatabase();

    //! Search a given directory for modules and register them
    void indexDirectory(fs::path pth);

    std::vector<fs::path> m_searchPaths; //!< Algorithm search paths
    std::vector<Algorithm::Info*> m_known; //!< List of known algorithms

    //! Stores loaded library handles
    std::map<std::string, void*> m_libraries;

    /**\briefStores libraries and their algorithms
     * 
     * When the key is NULL, the list holds built-in algorithms.
     */
    std::map<void*, std::vector<Algorithm*> > m_algos;

    //! List of algorithms built into the binary
    std::vector<std::pair<Algorithm::Info*, void*> > m_compiled;
};

};

#endif
