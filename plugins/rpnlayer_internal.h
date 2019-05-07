#ifndef TRT_RPNLAYER_INTERNAL_H
#define TRT_RPNLAYER_INTERNAL_H

#include "rpnlayer.h"
#include <cstdint>

// STRUCT BBOX {{{
template <typename TFloat>
struct Bbox
{
    TFloat x1, y1, x2, y2;
};

template <typename TFloat>
bool operator==(const Bbox<TFloat>& lhs, const Bbox<TFloat>& rhs)
{
    return lhs.x1 == rhs.x1 && lhs.y1 == rhs.y1 && lhs.x2 == rhs.x2 && lhs.y2 == rhs.y2;
}
// }}}

unsigned int hash(const void* array_, size_t size);

int8_t* alignPtr(int8_t* ptr, uintptr_t to);

int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize);

size_t calculateTotalWorkspaceSize(size_t* workspaces, int count);

#endif // TRT_RPNLAYER_INTERNAL_H