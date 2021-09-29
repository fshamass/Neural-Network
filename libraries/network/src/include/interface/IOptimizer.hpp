#ifndef __IOPTIMIZER__
#define __IOPTIMIZER__

#include <iostream>

class IOptimizer {
public:
    virtual void preUpdate() = 0;
    virtual void update(uint32_t layer) = 0;
    virtual void postUpdate() = 0;
};

#endif