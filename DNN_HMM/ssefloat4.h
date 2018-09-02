// ssematrix.h -- matrix with SSE-accelerated operations
//
// F. Seide, Nov 2010
//
// $Log: /Speech_To_Speech_Translation/dbn/dbn/ssefloat4.h $
// 
// 10    1/23/14 5:50p Fseide
// (changed a return type to avoid an unnecessary type conversion)
// 
// 9     1/23/14 5:12p Fseide
// (maskby now takes a generic 'int' as a template parameter to avoid type
// conversion--every cycle counts here, unfortunately)
// 
// 8     1/16/14 16:17 Fseide
// added a 'const' specifier to masktable[]
// 
// 7     1/15/14 18:59 Fseide
// moved masktable[] from unquantizeqbword1() into a new method
// float4::maskby(size_t) with truly static intialization of the table
// 
// 6     1/14/14 17:29 Fseide
// extensions for use in quantization
// 
// 5     3/03/11 8:16a Dongyu
// added weight sparseness support in training.
// 
// 4     11/22/10 9:53 Fseide
// (added a commented-out function)
// 
// 3     11/22/10 9:18 Fseide
// new method transpose(), for use in ssematrix::transpose()
// 
// 2     11/19/10 17:06 Fseide
// new method prefetch() (although not yet put to good use)
// 
// 1     11/12/10 13:25 Fseide
// moved out from ssematrix.h and cleaned up

#pragma once
#define _SSEFLOAT4_H        // this tells one of our consumers (from cudamatrix...) that we are defined (not so nice)

#include <intrin.h>         // for intrinsics
#include <stdlib.h>
namespace msra { namespace math {

// ===========================================================================
// float4 -- wrapper around the rather ugly SSE intrinsics for float[4]
//
// Do not use the intrinsics outside anymore; instead add all you need into this class.
//
// MSDN links:
// basic: http://msdn.microsoft.com/en-us/library/x5c07e2a%28v=VS.80%29.aspx
// load/store: (add this)
// newer ones: (seems no single list available)
// ===========================================================================

class float4
{
    __m128 v;   // value
private:
    // return the low 'float'
    float f0() const { float f; _mm_store_ss (&f, v); return f; }
    // construct from a __m128, assuming it is a f32 vector (needed for directly returning __m128 below)
    float4 (const __m128 & v) : v (v) {}
    // return as a __m128 --should this be a reference?
    operator __m128() const { return v; }
    // assign a __m128 (needed for using nested float4 objects inside this class, e.g. sum())
    float4 & operator= (const __m128 & other) { v = other; return *this; }
public:
    float4() {} // uninitialized
    float4 (const float4 & f4) : v (f4.v) {}
    float4 & operator= (const float4 & other) { v = other.v; return *this; }

    // construct from a single float, copy to all components
    float4 (float f) : v (_mm_load1_ps (&f)) {}
    //float4 (float f) : v (_mm_set_ss (f)) {}  // code seems more complex than _mm_load1_ps()

    // basic math
    float4 operator-() const { return _mm_sub_ps (_mm_setzero_ps(), v); }  // UNTESTED; setzero is a composite

    float4 operator& (const float4 & other) const { return _mm_and_ps (v, other); }
    float4 operator| (const float4 & other) const { return _mm_or_ps (v, other); }
    float4 operator+ (const float4 & other) const { return _mm_add_ps (v, other); }
    float4 operator- (const float4 & other) const { return _mm_sub_ps (v, other); }
    float4 operator* (const float4 & other) const { return _mm_mul_ps (v, other); }
    float4 operator/ (const float4 & other) const { return _mm_div_ps (v, other); }

    float4 & operator&= (const float4 & other) { v = _mm_and_ps (v, other); return *this; }
    float4 & operator|= (const float4 & other) { v = _mm_or_ps (v, other); return *this; }
    float4 & operator+= (const float4 & other) { v = _mm_add_ps (v, other); return *this; }
    float4 & operator-= (const float4 & other) { v = _mm_sub_ps (v, other); return *this; }
    float4 & operator*= (const float4 & other) { v = _mm_mul_ps (v, other); return *this; }
    float4 & operator/= (const float4 & other) { v = _mm_div_ps (v, other); return *this; }

    // warning: these do not return 'float4' values but 'int4' with values 0 and 0xffffffff
    float4 operator>= (const float4 & other) const { return _mm_cmpge_ps (v, other); }
    float4 operator<= (const float4 & other) const { return _mm_cmple_ps (v, other); }

    // returns 4 bits saying whether this is >= other
    // Bit n corresponds to the float that would be saved in a float array at offset [n].
    unsigned int comparesultasbits() const { return (unsigned int) _mm_movemask_ps (v); }        // 'this' must be result of operator>= or <=
    //int compare (const float4 & other) const { return _mm_movemask_ps (*this >= other); }

    // select from 'this' or 0, based on operator>= result (false -> 0; true -> this[k])
    // (operator>= returns an 'int4' that is either 0xffffffff or 0, so we can just use it here as a mask (int 0 has same bit representation as float 0))
    float4 maskby (const float4 & comparesult) const { return _mm_and_ps (comparesult, v); }    // 'comparesult' must be result of operator>= or <=
    static const __m128 masktable[16];      // [i] -> { i[0], i[1], i[2], i[3] }  where i is a 4-bit bit vector
    template<typename BITSASINT>            // (template to avoid type cast to size_t)
    float4 maskby (BITSASINT bits) const    // same, but 'mask' consists of 4 bits instead, as obtained by comparesultasbits()
    {
        assert (bits < _countof(masktable));
        float4 qval = masktable[bits];      // contains 1 for set bits and 0 for unset ones
        return qval * v;                    // component-wise product
    }

    // not yet implemented binary arithmetic ops: sqrt, rcp (reciprocal), rqsrt, min, max

    // other goodies I came across (intrin.h):
    //  - _mm_prefetch
    //  - _mm_stream_ps --store without polluting cache
    //  - unknown: _mm_addsub_ps, _mm_hsub_ps, _mm_movehdup_ps, _mm_moveldup_ps, _mm_blend_ps, _mm_blendv_ps, _mm_insert_ps, _mm_extract_ps, _mm_round_ps
    //  - _mm_dp_ps dot product! http://msdn.microsoft.com/en-us/library/bb514054.aspx
    //    Not so interesting for long vectors, we get better numerical precision with parallel adds and hadd at the end

    // prefetch a float4 from an address
    static void prefetch (const float4 * p) { _mm_prefetch ((const char *) const_cast<float4 *> (p), _MM_HINT_T0); }

    // transpose a 4x4 matrix
    // Passing input as const ref to ensure aligned-ness
    static void transpose (const float4 & col0, const float4 & col1, const float4 & col2, const float4 & col3,
                           float4 & row0, float4 & row1, float4 & row2, float4 & row3)
    {   // note: the temp variable here gets completely eliminated by optimization
        float4 m0 = col0; float4 m1 = col1; float4 m2 = col2; float4 m3 = col3;
        _MM_TRANSPOSE4_PS (m0, m1, m2, m3); // 8 instructions for 16 elements
        row0 = m0; row1 = m1; row2 = m2; row3 = m3;
    }

    // save a float4 to RAM bypassing the cache ('without polluting the cache')
    void storewithoutcache (float4 & r4) const
    {
        //_mm_stream_ps ((float*) &r4, v);
        r4 = v;
    }

#if 0
    // save a float4 to RAM bypassing the cache ('without polluting the cache')
    void storewithoutcache (float4 * p4) const
    {
        //_mm_stream_ps ((float*) p4, v);
        *p4 = v;
    }

    // save a float to RAM bypassing the cache ('without polluting the cache')
    void storewithoutcache (float & r) const
    {
        _mm_stream_ss (&r, v);
    }
#endif

    // return the horizontal sum of all 4 components
    // ... return float4, use another mechanism to store the low word
    float sum() const { float4 hsum = _mm_hadd_ps (v, v); hsum = _mm_hadd_ps (hsum, hsum); return hsum.f0(); }

    // please add anything else you might need HERE
};

// initialization of the lookup table for maskby(int):
#pragma push_macro ("M1")
#pragma push_macro ("M4")
#define M1(f) (f?1:0)
#define M4(n) { M1(n&1), M1(n&2), M1(n&4), M1(n&8) }
__declspec(selectany)   // allows us to compile this in multiple compilation units
const __m128 float4::masktable[16] = { M4(0), M4(1), M4(2), M4(3), M4(4), M4(5), M4(6), M4(7), M4(8), M4(9), M4(10), M4(11), M4(12), M4(13), M4(14), M4(15) }; // mask for each of the 16 value combinations
#pragma pop_macro ("M4")
#pragma pop_macro ("M1")

};};
