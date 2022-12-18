/*#-------------------------------------------------
#
#           std::string utils library
#
#    by AbsurdePhoton - www.absurdephoton.fr
#
#                v1.5 - 2022/12/12
#
#   - check if string is a number
#   - convert integer value to Hex
#   - operations on strings :
#       * upper and lower case
#       * clean string
#       * replacements
#       * deletion
#   - filename extractions :
#       * filename
#       * basename
#       * extension
#       * path
#
#-------------------------------------------------*/


#include "string-utils.h"


bool stringutils::isNumber(const std::string &st) // check if string is a number
{
    return std::regex_match(st, std::regex(( "((\\+|-)?[[:digit:]]+)(\\.(([[:digit:]]+)?))?")));
}

std::string stringutils::int2Hex(const int &number , const int &hex_len) // convert an integer number to string in hexadecimal
{
    std::ostringstream ss;
    ss << std::setfill('0') << std::setw(hex_len) << std::hex << number;

    return ss.str();
}

std::string stringutils::ToLower(std::string str) // get lowercase string
{
    std::string result = str;

    for (int n = 0; n < int(result.length()); n++)
        result[n] = std::tolower(result[n]);

    return result;
}

std::string stringutils::ToUpper(std::string str) // get uppercase string
{
    std::string result = str;

    for (int n = 0; n < int(result.length()); n++)
        result[n] = std::toupper(result[n]);

    return result;
}

std::string stringutils::CleanCharacter(const std::string &str, const char &chr) // clean leading, ending and double occurences of character in string
{
    std::string result = str;

    // ending spaces
    while (result[result.length() - 1] == chr)
        result.erase(result.length() - 1);
    // leading spaces
    while (result[0] == chr)
        result.erase(0, 1);
    // double spaces
    std::size_t find;
    while ((find = result.find(std::string(1, chr) + std::string(1, chr))) != std::string::npos)
       result.erase(find, 1);

    return result;
}

std::string stringutils::CleanCharacterBeforeAfter(const std::string &str, const char &chrToFind, const char &chrToDelete) // clean leading and ending character occurences next to character in string
{
    std::string result = str;

    std::size_t find;
    // before
    while ((find = result.find(std::string(1, chrToDelete) + std::string(1, chrToFind))) != std::string::npos)
       result.erase(find, 1);
    // after
    while ((find = result.find(std::string(1, chrToFind) + std::string(1, chrToDelete))) != std::string::npos)
       result.erase(find + 1, 1);

    return result;
}

std::string stringutils::ReplaceCharacter(const std::string &str, const char &chrToReplace, const char &chrReplace) // replace character by another
{
    std::string result = str;
    std::size_t find;

    while ((find = result.find(chrToReplace)) != std::string::npos)
       result[find] = chrReplace;

    return result;
}

std::string stringutils::ReplaceDecimalCharacter(const std::string &str) // replace decimal character by the one used in locale
{
    std::string result = str;
    std::size_t find;

    char point = std::use_facet< std::numpunct<char> >(std::cout.getloc()).decimal_point();
    if (point == ',') {
       find = result.find('.');
       result[find] = point;
    }

    return result;
}

std::string stringutils::DeleteAllCharacter(const std::string &str, const char &chrToDelete) // delete all occurences of character
{
    std::string result = str;
    std::size_t find;

    while ((find = result.find(chrToDelete)) != std::string::npos)
       result.erase(find, 1);

    return result;
}

std::string stringutils::DeleteNonPrintableCharacters(std::string str) // delete all values < 32 (space)
{
    std::string result = str;

    for (int n = 0; n < int(result.length()); n++) { // parse string
        if (result[n] < 32) { // non-printable character ?
            result.erase(n, 1); // delete it
            n--; // go back one character
        }
    }

    return result;
}

std::string stringutils::GetFolderFromFullPath(std::string str) // get folder from full file path
{
    return str.substr(0, str.find_last_of("/"));
}

std::string stringutils::GetFilenameFromFullPath(std::string str) // get filename from full file path
{
    return str.substr(str.find_last_of("/") + 1);
}

std::string stringutils::GetFilenameExtension(std::string str) // get filename extension
{
    return str.substr(str.find_last_of(".") + 1);
}

std::string stringutils::GetFilenameBasename(std::string str) // get filename basename
{
    std::string filename = GetFilenameFromFullPath(str);
    return str.substr(0, str.find_first_of(".") - 1);
}

