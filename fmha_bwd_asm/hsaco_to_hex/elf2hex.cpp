#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

using namespace std;

std::string toHex(char input)
{
    std::stringstream ss;
    ss << std::hex << std::uppercase;
    unsigned int tmp = static_cast<unsigned int>(input);
    if (tmp > 255)
        tmp = static_cast<unsigned char>(tmp);
    ss << "0x" << std::setw(2) << std::setfill('0') << tmp << ", ";
    return ss.str();
}

int main(int argc, char *argv[])
{
    std::ifstream is(argv[1], std::ios::binary | std::ios::ate);
    size_t nbytes = is.tellg();
    is.seekg(0, std::ios::beg);
    std::vector<char> buffer(nbytes, 0);
    is.read(&buffer[0], nbytes);

    std::ofstream outputFile;
    outputFile.open(argv[2]);
    if (outputFile.is_open())
    {
        for (int i = 0; i < nbytes; i++)
        {
            if (i != 0 && i % 16 == 0)
                outputFile << std::endl;
            outputFile << toHex(buffer[i]);
        }
    }
    else
    {
        std::cout << "Fail to open the file." << std::endl;
    }
    outputFile.close();

    return 0;
}

// g++ elf2hex.cpp
// ./a.out input.co output.hex