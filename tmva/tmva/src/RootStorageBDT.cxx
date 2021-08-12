#include "TMVA/RootStorageBDT.h"

namespace TMVA{
namespace RootStorage{

namespace INTERNAL{
     bool XMLAttributes::set(std::string const& name, std::string const& value) {
            if (name == "itree")
                return setValue(itree_, std::stoi(value));
            if (name == "boostWeight")
                return setValue(boostWeight_, std::stod(value));
            if (name == "pos")
                return setValue(pos_, value[0]);
            if (name == "depth")
                return setValue(depth_, std::stoi(value));
            if (name == "IVar")
                return setValue(IVar_, std::stoi(value));
            if (name == "Cut")
                return setValue(Cut_, std::stod(value));
            if (name == "res")
                return setValue(res_, std::stod(value));
            if (name == "purity")
                return setValue(purity_, std::stod(value));
            if (name == "nType")
                return setValue(nType_, std::stoi(value));
            return true;
        }

     bool XMLAttributes::hasValue(std::string const& name){
            if (name == "itree")
                return itree_.has_value();
            if (name == "boostWeight")
                return boostWeight_.has_value();
            if (name == "pos")
                return pos_.has_value();
            if (name == "depth")
                return depth_.has_value();
            if (name == "IVar")
                return IVar_.has_value();
            if (name == "Cut")
                return Cut_.has_value();
            if (name == "res")
                return res_.has_value();
            if (name == "purity")
                return purity_.has_value();
            if (name == "nType")
                return nType_.has_value();
            return false;
        }

    void XMLAttributes::reset(){
        boostWeight_.reset();
        itree_.reset();
        pos_.reset();
        depth_.reset();
        IVar_.reset();
        Cut_.reset();
        res_.reset();
        purity_.reset();
        nType_.reset();
    }

    template <class T>
    bool XMLAttributes::setValue(std::optional<T>& member, T const& value) {
        if (member.has_value()) {
            member = value;
            return false;
        }
        member = value;
        return true;
    }
}
BDT::Parse(std::string filepath, bool use_purity){

    char sep = '/';
    #ifdef _WIN32
    sep = '\\';
    #endif

    size_t isep = filepath.rfind(sep, filepath.length());
    if (isep != std::string::npos){
      filename = (filepath.substr(isep+1, filename.length() - isep));
    }

    //Check on whether the TMVA BDT XML file exists
    if(!std::ifstream(filepath).good()){
        throw std::runtime_error("Model file "+filename+" not found!");
    }

    std::ifstream fPointer(filepath);
    std::string xmlString;
    fPointer.seekg(0, std::ios::end);
    xmlString.reserve(fPointer.tellg());
    fPointer.seekg(0, std::ios::beg);
    xmlString.assign((std::istreambuf_iterator<char>(fPointer)), std::istreambuf_iterator<char>());

    INTERNAL::BDTWithXMLAttributes bdtXmlAttributes;
    std::vector<INTERNAL::XMLAttributes>* currentTree = nullptr;
    INTERNAL::XMLAttributes* attrs = nullptr;




}

}
}
