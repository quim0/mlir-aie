#include <aie_api/aie.hpp>

#define TILE_WIDTH 32
#define VECT_FACTOR 16

extern "C" {
    void alignment_test_kernel(int32_t* bufin, int32_t* bufout) {
        auto to_cmp = aie::broadcast<int32_t, VECT_FACTOR>(1);
        for (int i=0; i<TILE_WIDTH; i+=VECT_FACTOR) {
            aie::vector<int32_t, VECT_FACTOR> vectin = aie::load_v(bufin + i);
            auto result = aie::add(vectin, to_cmp);
            aie::store_v(bufout + i, result);
        }
    }
}