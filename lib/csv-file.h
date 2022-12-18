/*#-------------------------------------------------
#
#                 CSV file library
#
#    by AbsurdePhoton - www.absurdephoton.fr
#
#                v1.0 - 2020/06/10
#
#  Open file, read lines and get values, close file
#
#-------------------------------------------------*/

#ifndef CSVFILE_H
#define CSVFILE_H

#include <fstream>
#include <vector>

#include "string-utils.h"

class csvfile
{
public:
    static bool OpenFile(std::ifstream &csvFile, const std::string &csvFilename); // open csv file to read - return success or not
    static bool ReadLine(std::ifstream &csvFile, std::vector<std::string> &data, const char &separator); // read one line from csv file and store text fields in vector
    static void CloseFile(std::ifstream &csvFile); // close csv file
};

#endif // CSVFILE_H
