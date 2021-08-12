// @(#)root/tmva $Id$
// Author: Jonas Rembser, Sanjiban Sengupta, 2021

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : RootStorageBDT                                                        *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *     Class for storing trained BDT models into Root files                       *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Jonas Rembser                                                             *
 *      Sanjiban Sengupta <sanjiban.sg@gmail.com>                                 *
 *                                                                                *
 * Copyright (c) 2021:                                                            *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef TMVA_SOFIE_ROOTSTORAGEBDT
#define TMVA_SOFIE_ROOTSTORAGEBDT

#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "TBuffer.h"


namespace TMVA{
namespace RootStorage{

namespace INTERNAL{
    struct BDTWithXMLAttributes {
        std::vector<double> boostWeights;
        std::vector<std::vector<XMLAttributes>> nodes;
        };

    class XMLAttributes {
        private:
        template <class T>
        bool setValue(std::optional<T>& member, T const& value);
        // from the tree root node node
        std::optional<double> boostWeight_ = std::nullopt;
        std::optional<int> itree_ = std::nullopt;
        std::optional<char> pos_ = std::nullopt;
        std::optional<int> depth_ = std::nullopt;
        std::optional<int> IVar_ = std::nullopt;
        std::optional<double> Cut_ = std::nullopt;
        std::optional<double> res_ = std::nullopt;
        std::optional<double> purity_ = std::nullopt;
        std::optional<int> nType_ = std::nullopt;

        public:
        // If we set an attribute that is already set, this will do nothing and return false.
        // Therefore an attribute has repeated and we know a new node has started.
        bool set(std::string const& name, std::string const& value);
        bool hasValue(std::string const& name);
        void reset();
        auto const& itree() const { return itree_; };
        auto const& boostWeight() const { return boostWeight_; };
        auto const& pos() const { return pos_; };
        auto const& depth() const { return depth_; };
        auto const& IVar() const { return IVar_; };
        auto const& Cut() const { return Cut_; };
        auto const& res() const { return res_; };
        auto const& purity() const { return purity_; };
        auto const& nType() const { return nType_; };
    };
}

class BDT: public TObject{
    private:
    struct SlowTreeNode {
        bool isLeaf = false;
        int depth = -1;
        int index = -1;
        int yes = -1;
        int no = -1;
        int missing = -1;
        int cutIndex = -1;
        double cutValue = 0.0;
        double leafValue = 0.0;
        };
    std::vector<SlowTreeNode> SlowTree;
    std::vector<SlowTree> SlowForest;
    std::string filename;

    public:
    void Parse(std::string filepath, bool use_purity=true);

    ClassDef(BDT,1);
  };
}//RootStorage
}//TMVA

#endif //TMVA_ROOTSTORAGE_BDT
