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

size_t charToIndex(char key);
int saveTrainingElement(oglopp::SSBO& buffer, uint8_t key, std::string const& parentDir);
void loadTrainingFiles(std::vector<std::vector<float>>& files, std::vector<uint32_t>& fileIndices, std::string const& parentDir);
void setExpectedOutput(Network& network);
void doSomeSamples(oglopp::Compute& compute, Network& network, std::string const& parentDir, std::vector<std::vector<float>>& files, std::vector<uint32_t>& fileIndices, size_t& offset, size_t countToDo);

/* @brief Read some layer of some type and return the type that was read for safe keeping
 * @param[in] stream	A reference to the input stream to read the layer from
 * @param[in] network	A reference to the netwrok to push layers to
*/
Layer::Type readLayer(std::fstream& stream, Network& network);

/* @brief Write some layer of some type and return the type that was written for safe keeping
 * @param[in] stream	A reference to the input stream to read the layer from
 * @param[in] layer		A pointer to the layer to write
*/
Layer::Type writeLayer(std::fstream& stream, Layer& layer);

#endif
