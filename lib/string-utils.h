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

#ifndef STRINGUTILS_H
#define STRINGUTILS_H


#include <string>
#include <regex>
#include <iostream>
#include <iomanip>


class stringutils
{
public:
    // numbers
    static bool isNumber(const std::string &st); // check if string is a number
    static std::string int2Hex(const int &number , const int &hex_len); // convert an integer number to string in hexadecimal
    // case
    static std::string ToLower(std::string str); // get lowercase string
    static std::string ToUpper(std::string str); // get uppercase string
    // cleaning
    static std::string CleanCharacter(const std::string &str, const char &chr); // clean leading, ending and double occurences of character in string
    static std::string CleanCharacterBeforeAfter(const std::string &str, const char &chrToFind, const char &chrToDelete); // clean leading and ending character occurences next to character in string
    // common operations
    static std::string ReplaceCharacter(const std::string &str, const char &chrToReplace, const char &chrReplace); // replace character by another
    static std::string ReplaceDecimalCharacter(const std::string &str); // replace decimal character by the one used in locale
    static std::string DeleteAllCharacter(const std::string &str, const char &chrToDelete); // delete all occurences of character
    static std::string DeleteNonPrintableCharacters(std::string str); // delete all non-printable characters
    // filename extractions
    static std::string GetFolderFromFullPath(std::string str); // get folder from full file path
    static std::string GetFilenameFromFullPath(std::string str); // get filename from full path
    static std::string GetFilenameExtension(std::string str); // get filename extension
    static std::string GetFilenameBasename(std::string str); // get filename basename
};

#endif // STRINGUTILS_H
