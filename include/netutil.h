#ifndef NETUTIL_H
#define NETUTIL_H

#include "defines.h"
#include "network.h"

#include <cstddef>
#include <iostream>
#include <filesystem>
#include <oglopp/ssbo.h>
#include <oglopp/window.h>
#include <oglopp/compute.h>
#include <fstream>
#include <vector>

#define SAMPLES_DIR	"samples/"

size_t charToIndex(char key);
int saveTrainingElement(oglopp::SSBO& buffer, uint8_t key, std::string const& parentDir);
void loadTrainingFiles(std::vector<std::vector<float>>& files, std::vector<uint32_t>& fileIndices, std::string const& parentDir);
void setExpectedOutput(Network& network);
void doSomeSamples(oglopp::Compute& compute, Network& network, std::string const& parentDir, std::vector<std::vector<float>>& files, std::vector<uint32_t>& fileIndices, size_t& offset, size_t countToDo);

#endif
