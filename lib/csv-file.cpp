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

#include "csv-file.h"


bool csvfile::OpenFile(std::ifstream &csvFile, const std::string &csvFilename) // open csv file to read - return success or not
{
    csvFile.open(csvFilename, std::ifstream::in); // open csv file

    return csvFile.is_open(); // success or not
}

bool csvfile::ReadLine(std::ifstream &csvFile, std::vector<std::string> &fields, const char &separator) // read one line and store text fields in vector
{
    std::string line; // line to read from file
    fields.clear();

    while ((csvFile) and ((line.length() == 0) or (line[0] == 13) or (line[0] == '#'))) // not at the end of file and line is not empty
        std::getline(csvFile, line); // get another line

    if (!csvFile) // end of file
        return false;

    //line = stringutils::DeleteNonPrintableCharacters(line); // delete unwanted special characters
    line = stringutils::CleanCharacter(line, char(13)); // replace windows/mac end of line
    line = stringutils::ReplaceCharacter(line, char(9), char(' ')); // replace tab characters by spaces
    line = stringutils::CleanCharacter(line, ' '); // clean leading and ending spaces, and double spaces

    if (line[0] == '#') // comment
        return true; // comment : it's not an error but vector is untouched

    fields.clear(); // clear fields fields
    int pos = 0; // init character position
    int old_pos = 0; // save field's first character position
    while (pos < int(line.length())) { // parse text line
        if (line[pos] == '"') { // if first character is a double-quote
            pos++; // update position
            while ((pos < int(line.length())) and (line[pos] != char('"'))) { // find next double-quote
                if ((pos + 1 < int(line.length())) and (line[pos] == char('"'))) // if double-quote is doubled
                    pos = pos +2; // pass it
                else // not a doubled double-quote
                    pos++; // next character
            }
        }

        while ((pos < int(line.length())) and (line[pos] != separator)) // find next separator character
            pos++; // update current character position

        std::string field = line.substr(old_pos, pos - old_pos); // extract field text
        field = stringutils::CleanCharacter(field, ' '); // clean leading and ending spaces, and double spaces
        field = stringutils::CleanCharacter(field, '"'); // clean leading and ending double-quotes, and double double-quotes
        fields.push_back(field); // store field value
        pos++; // step over semicolon character
        old_pos = pos; // new first character position
    }
    if (line[line.length() - 1] == separator) { // if last character is a separator
        std::string field = ""; // empty value
        fields.push_back(field); // store empty value
    }

    return true;
}

void csvfile::CloseFile(std::ifstream &csvFile) // close csv file
{
    csvFile.close(); // close csv file
}
