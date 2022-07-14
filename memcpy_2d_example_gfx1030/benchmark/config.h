#ifndef CONFIG_
#define CONFIG_

#include <unordered_map>
#include <random>

template <typename T>
class Matrix_2d
{
public:
    int rows;
    int cols;
    int padding;
    int length;
    bool type; //Host: 1; Device: 0
    T *data;
public:
    Matrix_2d(): rows(0), cols(0), padding(0), length(0), type(0), data(nullptr) {}
    Matrix_2d(int r, int c, int p, bool t): rows(r), cols(c), padding(p), type(t), length(r * (c + p)), data(new T[r * (c + p)]) {}
    ~Matrix_2d() {
        if (data != nullptr && type) {
            delete [] data;
        }
    }
    void initMem() {
        if (!data)
            data = new T[rows * (cols + padding)];
    }
};

class Config
{
public:
    char m_delimiter;
    std::unordered_map<std::string, std::string> m_contents;
public:
	struct FILE_NOT_FOUND 
    {
		std::string filename;
		FILE_NOT_FOUND(const std::string& filename_ = std::string()): filename(filename_) {}
	};
public:
    Config(): m_delimiter('=') {}
    void parseConfigFile(const char *fileName);
    void split(const std::string &line);
private:
    void deleteSpace(std::string &str);
};  


#endif 