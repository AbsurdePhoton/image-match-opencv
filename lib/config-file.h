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

#ifndef CONFIGFILE_H
#define CONFIGFILE_H

#include <fstream>
#include <vector>

#include "string-utils.h"

// dont forget to set your locales to use '.' as decimal separator ! use "std::setlocale(LC_NUMERIC,"C");" at the beginning of your program

class configfile
{
public:
    struct struct_config {
        std::string type; // entry type : object, comment, value, unknown
        std::string name; // entry name
        int index; // entry index
        std::string valueType; // value : text, integer or decimal, boolean
        std::vector<std::string> valuesText;
        std::vector<double> valuesNumber;
        bool valueBool;
    };

    static bool OpenFile(std::ifstream &configFile, const std::string &configFilename); // open config file to read - return success or not
    static bool ReadLine(std::ifstream &configFile, struct_config &configValues); // read one line and store values
    static void CloseFile(std::ifstream &configFile); // close config file
};

#endif // CONFIGFILE_H

/*

To open and read a config file :
    - use configfile::Openfile with a std::ifstream object, returns a boolean to indicate success or not
    - read a line with configfile::FReadLine :
        * returns false if end of file
        * if line was read returns true, values are in a struct_config object - do what you want with the values, explanation below
        * "number" values are always stored as double, use casting to get the desired type in your code (e.g. int(myobject.valuesNumber[0] ). In case of a decimal number, use a "." as decimal separator
        *  "text" values should always be between quotation marks ' " ' -- anyway if a string doesn't have a digit at the end, it is considered as text
        *  text and number values are stored in a std::vector - there can be multiple values separated by a comma ","
        *  comment have a "#' at the beginning
    - close the file with configfile::CloseFile

Structure of an example config file :

# beginning of example      <- this a comment starting with "#"
[hatches00]                 <- line starts with a "[" this is an object : type = "object", name="hatches", index = 0
    name="Colored pencil"   <- this is a value : type="value", name="name", type="text", content is in valuesText="Colored pencil"
    posFuzzing=2            <- this is a value : type="value", name="posFuzzing", valueType="number", content is in valuesNumber=2
    angleFuzzing=0.5
    # this is acomment      <- line starts with "#" : comment
    [hatchesItem]           <- an object can contain another object
        name="default"      <- this object's name
        type="hatches"      <- here this a value, the object type can't be set in config file
    [/hatchesItem]          <- an object must be closed by "/objectname"
    [hatchesItem]           <- another object, etc
        name="100%"
        grayLevels=0,100    <- values can be a list of numbers, separated by commas
        angle=60
        spacing=2
        shift=0
        texts="one", "two"  <- example of multiple text values
    [/hatchesItem]
[/hatches]                  <- always close an object !
# end of example config file

You can indent as you want, no control is done if the indentation is correct - only use <tab> and <space>

If a line can't be read or contains garbage, or has a syntax error :
    - type ="error"
    - name ="cause of error"

*/
