#include <iostream>
#include <string>
#include <fstream>

#include "config.h"

void Config::parseConfigFile(const char *fileName)
{
    std::ifstream in(fileName);
    if (!in)
        throw FILE_NOT_FOUND();
    
    std::string line;
    while (!in.eof()) {
        getline(in, line);
        if (line.find('[') == 0 || line.find(' ') == 0)
            continue;
        split(line);
    }
}

void Config::split(const std::string &line)
{   
    std::size_t previous = 0;
    std::size_t current = line.find(this->m_delimiter);
    std::size_t length = line.length();
    if (current != std::string::npos) {
        std::string key = line.substr(previous, current - previous);
        std::string value = line.substr(current + 1, length);
        deleteSpace(key);
        deleteSpace(value);    
        this->m_contents[key] = value;
    }
}

void Config::deleteSpace(std::string &str)
{
    int j = 0;
    for (int i = 0; i < str.length(); i++) {
        if (str[i] == ' ')
            continue;
        str[j++] = str[i];
    }
    str.resize(j);
}

