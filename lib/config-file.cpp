/*#-------------------------------------------------
#
#               Config file library
#
#    by AbsurdePhoton - www.absurdephoton.fr
#
#                v1.0 - 2020/06/24
#
#  Open file, read lines and get values, close file
#
#-------------------------------------------------*/

#include <algorithm>

#include "config-file.h"


bool configfile::OpenFile(std::ifstream &configFile, const std::string &configFilename) // open config file to read - return success or not
{
    configFile.open(configFilename, std::ifstream::in); // read config file

    return configFile.is_open(); // return success or not
}

bool configfile::ReadLine(std::ifstream &configFile, struct_config &configValues) // read one line of config file and extract values
{
    std::string line; // line to read from file
    while ((configFile) and ((line.length() == 0) or ((line.length() == 1) and (line[0] == 13)))) { // not at the end of file and line is not empty
        std::getline(configFile, line);
    }

    if ((!configFile) or (line.length() == 0)) // end of file or line is empty
        return false;

    //// clean line
    //line = stringutils::DeleteNonPrintableCharacters(line); // delete unwanted special characters
    line = stringutils::CleanCharacter(line, char(13)); // replace windows/mac end of line
    line = stringutils::DeleteAllCharacter(line, ']'); // delete closing brackets
    line = stringutils::ReplaceCharacter(line, char(9), char(' ')); // replace tab characters by spaces
    line = stringutils::CleanCharacter(line, ' '); // clean leading and ending spaces, and double spaces
    line = stringutils::CleanCharacterBeforeAfter(line, '=', ' '); // clean spaces before and after equal symbol
    line = stringutils::CleanCharacterBeforeAfter(line, '[', ' '); // clean spaces before and after bracket symbol
    line = stringutils::CleanCharacterBeforeAfter(line, ']', ' '); // clean spaces before and after bracket symbol
    line = stringutils::CleanCharacterBeforeAfter(line, '#', ' '); // clean spaces before and after hash symbol

    //// type
    switch (line[0]) {
        case '[':
            configValues.type = "object";
            break;
        case '#':
            configValues.type = "comment";
            configValues.name = line.substr(1);
            return true;
        default: // unknown type
            if (isalpha(line[0])) {
                configValues.type = "value";
                break;
            } else {
                configValues.type = "unknown";
                return true;
            }
    }

    //// default values
    configValues.name = "none";
    configValues.index = -1;
    configValues.valueType = "none";
    configValues.valuesNumber.clear();
    configValues.valuesText.clear();

    //// malformed object
    if (((configValues.type == "object") and ((!isalpha(line[1])) and (line[1] != '/')))
            or ((configValues.type == "value") and (!isalpha(line[0])))) { // first character must be alphanumeric
        configValues.type = "error";
        configValues.name = "incorrect object or value name";
        return true;
    }

    std::size_t equalFound = line.find('='); // find position of equal character in line

    //// line is NOT a value -> object
    if (equalFound == std::string::npos) { // equal not found -> it's not a value -> it's an object
        std::string object;
        //// end of object
        if ((line.length() >= 2) and (line[1] == '/')) {
            configValues.type = "end-object";
            object = line.substr(2, std::string::npos); // copy string till the end
        }
        else
        //// define an object
            object = line.substr(1, std::string::npos); // copy string till the end
        object = stringutils::DeleteAllCharacter(object, ' ');

        //// object has an index
        if (isdigit(object[object.length() - 1])) { // if last character is a digit object has an index
            int pos = object.length() - 1; // from the end of the line
            while ((isdigit(object[pos]))) // parse string
                pos--; // a long as current character is a digit
            configValues.name = stringutils::ToLower(object.substr(0, pos + 1)); // get object name
            std::string index = object.substr(pos + 1, std::string::npos); // object index
            if ((stringutils::isNumber(index)) and (index.find('.') == std::string::npos)) // is the index really numeric ? it must be an integer (not dot inside the number)
                configValues.index = std::stoi(index); // it is numeric without a dot -> get object index
            else {
                configValues.type = "error"; // there's something else inside
                configValues.name = "object index error";
            }
        }
        //// object has no index
        else { // end of object name is not numeric
            configValues.name = stringutils::ToLower(object); // get object name
            configValues.index = -1; // no index
        }

        return true;
    }
    //// line is a value
    else if ((equalFound) and (equalFound < line.length() - 1) and (isalpha(line[0]))) { // equal found and is not at the end of the line -> it's a value
        // get value name and index
        std::string object = line.substr(0, equalFound); // copy string to the end
        object = stringutils::DeleteAllCharacter(object, ' '); // no space in value name

        //// value name has an index
        if (isdigit(object[object.length() - 1])) { // is last character a digit ?
            int pos = object.length() - 1; // from the end of the string
            while ((isdigit(object[pos]))) // find next digit character
                pos--; // as long as current character is a digit
            configValues.name = stringutils::ToLower(object.substr(0, pos + 1)); // get value name
            std::string index = object.substr(pos + 1, std::string::npos); // object index
            if ((stringutils::isNumber(index)) and (index.find('.') == std::string::npos)) // is the index really numeric ? it must be an integer (not dot inside the number)
                configValues.index = std::stoi(index); // it is numeric without a dot -> get object index
            else {
                configValues.type = "error"; // there's something else inside
                configValues.name = "value index error";
                return true;
            }
        }
        //// value name has no index
        else { // last character is not a digit
            configValues.name = stringutils::ToLower(object); // get value name
            configValues.index = -1; // no index
        }

        //// get values
        std::string value = line.substr(equalFound, std::string::npos); // get value in string

        //// error : no values after "equal" !!!
        if (!(value.length() > 0)) {
            configValues.type = "error"; // it's an empty value
            configValues.name = "value empty";
            return true;
        }

        //// values are numbers
        if (isdigit(value[value.length() - 1])) { // is last character a number ?
            value = stringutils::DeleteAllCharacter(value, ' ');
            int pos = value.length() - 1;
            while (value[pos] != '=') { // parse string until "equal" is found
                while ((isdigit(value[pos])) or (value[pos] == '.') or (value[pos] == 'e') or (value[pos] == '-')) // while character is part of a number
                   pos--; // change index position
                std::string val = value.substr(pos + 1); // get string value from pos+1 to the end
                configValues.valuesNumber.push_back(std::stod(val)); // get number value, add it to list

                if (value[pos] == ',') { // is it a list ?
                    value.erase(pos); // delete end of string + list separator
                    pos--; // next value
                }
                else if (value[pos] == '=') {
                    value.erase(pos + 1); // delete end of string but NOT the equal character
                }
                else { // not a list and not "equal" -> it's an error
                    configValues.type = "error";
                    configValues.name = "number value list error";
                    return true;
                }
            }
            std::reverse(configValues.valuesNumber.begin(), configValues.valuesNumber.end()); // values were read from the end
            configValues.valueType = "number";
            return true;
        }
        //// string values
        else { // value is a string
            value = stringutils::CleanCharacter(value, '"'); // delete beginning and ending and double quotes if any
            //// is it a boolean ?
            if ((stringutils::ToLower(value) == "=true") or (stringutils::ToLower(value) == "=false")) { // is text value a boolean ?
                configValues.valueBool = (stringutils::ToLower(value) == "=true"); // get boolean value
                configValues.valueType = "bool";
            }
            else { // it's not a boolean -> text values
                int pos = value.length() - 1; // set index to the end of string
                while (value[pos] != '=') { // parse until "equal" is found
                    while ((value[pos] != ',') and (value[pos] !='='))
                        pos--;
                    std::string val = value.substr(pos + 1); // get string value from pos+1 to the end
                    val = stringutils::CleanCharacter(val, '"'); // in case there are quotes
                    val = stringutils::CleanCharacter(val, ' '); // in case there are spaces
                    configValues.valuesText.push_back(val); // get string value

                    if (value[pos] == ',') { // is it a list ?
                        value.erase(pos); // delete end of string + list separator
                        pos--; // next value
                    }
                    else if (value[pos] == '=') {
                        value.erase(pos + 1); // delete end of string but NOT the equal character
                    }
                    else { // not a list and not "equal" -> it's an error
                        configValues.type = "error";
                        configValues.name = "value list error";
                        return true;
                    }
                }
                std::reverse(configValues.valuesText.begin(), configValues.valuesText.end()); // values were read from the end
                configValues.valueType = "text";
            }
            return true;
        }
    }
    else {
        configValues.type = "error"; // there's something else inside
        configValues.name = "malformed line";
        return true;
    }

    return true; // if we're here nothing was found
}

void configfile::CloseFile(std::ifstream &configFile) // close config file
{
    configFile.close(); // close config file
}
