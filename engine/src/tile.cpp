#include "scribblez/tile.h"

namespace scribblez {

// A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
const std::array<int, 26> LETTER_VALUES = {
    1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3,
    1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10};

// A B C D E F G H I J K L M N O P Q R S T U V W X Y Z, blank
const std::array<int, 27> TILE_COUNTS = {
    9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2,
    6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1, 2};

}  // namespace scribblez
