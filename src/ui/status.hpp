#ifndef STATUS_HPP
#define STATUS_HPP

#include <string>
#include <vector>
#include <map>
#include <utility>

namespace ui {

/**\brief Base class for dynamic text elements
 */
class Field {
public:
    Field();

    /**\brief Update the field to the newest available data
     */
    virtual void update()=0;

    /**\brief Get the field's native width (in characters)
     *
     * If the field expands to fit available space, this returns -1.
     */
    virtual int getNativeWidth() const=0;
    
    /**\brief Get a new string containing the field value
     *
     * Subclasses may want to implement more efficient versions, but a basic
     * version is provided automatically.
     *
     * \param width Padding target. If zero, ignored.
     */
    virtual std::string operator()(int width=0) const;

    /**\brief Get a C string containing the field value.
     *
     * The string is inherited by the caller.
     */
    virtual char* c_str(int width=0) const=0;

    /**\brief Return the field's update counter
     *
     * Each field contains a monotonically increasing update counter which
     * increments whenever it's changed. Clients can redraw only on changes by
     * storing the last known update counter value.
     */
    unsigned long getLastUpdate() const;

    /**\brief Increment the field's reference count
     */
    void ref();

    /**\brief Decrement the field's reference count
     */
    void unref();

    /**\brief Return the field's reference count
     */
    int refs();

protected:
    /**\brief Mark the field as updated
     */
    void notify();

private:
    unsigned long m_update;
    unsigned int m_refs;
};

/**\brief Utility class for holding and reusing fields
 */
class TUIManager {
public:
    TUIManager();
    ~TUIManager();

    /**\brief Register a new field with the given formatting code
     */
    void registerField(std::string name, Field* field);

    /**\brief Register a new field with the given formatting code and short code
     */
    void registerField(std::string name, char code, Field* field);

    /**\brief Update all in-use fields
     */
    void update();

    /**\brief Get a field for some static text
     */
    const Field* get_text(const std::string& text);

    /**\brief Get a field by formatting code
     */
    const Field* get(const std::string& name);

    /**\brief Get a field by short code
     */
    const Field* get(char code);

    /**\brief Release a field reference
     */
    void release(const Field* f);

private:
    std::map<std::string, Field*> m_name_map;
    std::map<char, Field*> m_short_map;
    std::map<std::string, Field*> m_static_map;
    std::vector<Field*> m_dynamic_fields;
};

/**\brief Represents a formatted string with some number of status fields in it
 */
class StatusLine {
public:
    StatusLine(std::string templ, TUIManager* mgr);
    ~StatusLine();

    const std::string& render();

private:
    void forceRender();

    std::string m_contents;
    TUIManager* m_manager;
    std::vector<std::pair<std::pair<unsigned long, int>, const Field*> > m_fields;
};

/**\brief Field showing static text
 */
class StaticField : public Field {
public:
    StaticField(std::string s);
    ~StaticField();

    int getNativeWidth() const;
    void update();
    std::string operator()(int width) const;
    char* c_str(int width) const;

private:
    std::string m_body;
};

/**\brief Base class for numeric fields
 */
class FloatField : public Field {
public:
    FloatField(float alpha=0.0);
    ~FloatField();

    float getValue();

    /**\brief Set the alpha constant used in this field's EMA
     */
    void setAlpha(float a);

    int getNativeWidth() const;
    char* c_str(int width) const;
    std::string operator()(int width) const;

protected:
    void update(float v);

private:
    float m_val;
    float m_alpha;
};

/**\brief Field showing the current CPU usage
 */
class CPULoad : public FloatField {
public:
    CPULoad();

    void update();

private:
    long m_last_total, m_last_me;
};

/**\brief Field showing an arbitrary user value
 */
class ValueField : public FloatField {
public:
    void addSample(float val);

    void update();
};

};

#endif
