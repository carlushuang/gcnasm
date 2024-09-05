#include <stdlib.h>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <cstdio>
#include <array>
#include <sstream>
#include <algorithm> // for std::remove
#include <map>

namespace fs = std::filesystem;

std::vector<std::string> execute_ls_command(std::string floder_asm)
{
    std::vector<std::string> output;
    std::string cmd = "ls " + floder_asm + "/*.s";
    // std::cout << "cmd:" << cmd << std::endl;
    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe)
    {
        std::cerr << "Failed to open pipe to ls command!" << std::endl;
        return output;
    }

    const size_t buffer_size = 128;       // Buffer size for a single line of output
    std::array<char, buffer_size> buffer; // Buffer for fgets()
    while (fgets(buffer.data(), buffer_size, pipe) != nullptr)
    {
        output.emplace_back(buffer.data());
    }

    pclose(pipe);
    return output;
}

void removeTrailingNewline(std::string &str)
{
    size_t pos = str.find_last_not_of("\r\n");
    if (pos != std::string::npos)
    {
        str.erase(pos + 1);
    }
    else
    {
        str.clear();
    }
}

std::map<std::string, std::string> parse_options(const std::vector<std::string> &optionList)
{
    std::map<std::string, std::string> options;
    for (const std::string &option : optionList)
    {
        size_t equalPos = option.find('=');
        if (equalPos != std::string::npos)
        {
            std::string key = option.substr(0, equalPos);
            std::string value = option.substr(equalPos + 1);
            // options[key] = std::stoull(value);
            options[key] = value;
        }
        else
        {
            // Handle error or ignore options with no equal sign
            std::cerr << "Error: Invalid option format: " << option << std::endl;
        }
    }
    return options;
}

void get_param(std::map<std::string, std::string> parsedOptions, std::string key, std::string &value)
{
    auto it = parsedOptions.find(key);
    if (it != parsedOptions.end())
    {
        value = it->second;
    }
}

int main(int argc, char *argv[])
{
    std::stringstream ss_elf2hex;
    ss_elf2hex << "g++ elf2hex.cpp -o elf2hex";
    system(ss_elf2hex.str().c_str());

    std::vector<std::string> options;
    for (int i = 1; i < argc; i++)
    {
        options.push_back(argv[i]);
    }
    std::map<std::string, std::string> parsedOptions = parse_options(options);
    std::string folder_asm = "../shaders";
    std::string folder_co = "shaders_co";
    std::string folder_hex = "shaders_hex";
    std::string cmd_elf2hex = "./elf2hex";
    get_param(parsedOptions, "in", folder_asm);
    get_param(parsedOptions, "co", folder_co);
    get_param(parsedOptions, "hex", folder_hex);
    get_param(parsedOptions, "cmd", cmd_elf2hex);

    std::cout << "cmd_elf2hex:" << cmd_elf2hex << std::endl;

    fs::path dir_path_co(folder_co);
    fs::path dir_path_hex(folder_hex);


    if (fs::create_directory(folder_co))
    {
        std::cout << "Folder co created successfully." << std::endl;
    }
    else
    {
        std::cout << "Failed to create folder co." << std::endl;
    }

    if (fs::create_directory(folder_hex))
    {
        std::cout << "Folder hex created successfully." << std::endl;
    }
    else
    {
        std::cout << "Failed to create folder hex." << std::endl;
    }

    std::vector<std::string> ls_output = execute_ls_command(folder_asm);
    for (std::string &line : ls_output)
    {
        removeTrailingNewline(line);
        std::string cut = line.substr(10, line.find_last_of(".") - 10);

        std::stringstream ss;
        std::string command = "/opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn--amdhsa --offload-arch=gfx942 ";
        ss << command << line << " -o " << folder_co << cut << ".co ";
        // std::cout << ss.str().c_str() << std::endl;
        system(ss.str().c_str());

        std::stringstream ss_hex;
        ss_hex << cmd_elf2hex << " " << folder_co << cut << ".co " << folder_hex << cut << ".hex";
        // std::cout << ss_hex.str().c_str() << std::endl;
        system(ss_hex.str().c_str());
    }
}
// g++ -std=c++17 gen_hex_shell.cpp
// ./a.out -co=shader_co -hex=shaders_hex -cmmd=./elf2hex -in=../shaders