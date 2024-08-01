/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */

#pragma once
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif

#pragma once
#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#endif
#endif
#endif


#define ANGLE 0.0245436926

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
#else
#include <x86intrin.h>
#endif
#endif

#include <algorithm>
#include <limits>
#include <vector>
#pragma once
#include "inttypes.h"
#include "stdio.h"

//_mm512_set_epi8 std::vector<uint32_t> popcnt_lookup_table = {
__m256i _popcnt_lookup_table_256 =    _mm256_set_epi8(
    4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0,
    4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0
);

__m512i _popcnt_lookup_table_512 = _mm512_castsi256_si512(_popcnt_lookup_table_256);
__m512i _popcnt_lookup_table = _mm512_inserti64x4(_popcnt_lookup_table_512, _popcnt_lookup_table_256, 1); 

__m512i _cos_select;// = _mm512_set1_epi32(63);
__m512 _cos_table0, _cos_table1, _cos_table2, _cos_table3, _cos_table4, _cos_table5, _cos_table6, _cos_table7;
int ultimate_select;



//const __m512i _popcnt_lookup_table = _mm512_loadu_si512(popcnt_lookup_table.data());

namespace hnswlib {


    template<typename dist_t> 
    struct Finger {

        int low_rank;
        int dimension;
        // uint32_t num_codebooks;
        //float scale;
        //float bias;
        int select;
        std::vector<float> projection_matrix;
        std::vector<float> codebook;

        void *dist_func_param_{nullptr};
        DISTFUNC<dist_t> fstdistfunc_;
        DISTFUNC<dist_t> finnerpfunc_;
        // no need to save, calculate in preprosee
        //pecos::bnn::HNSW<float, FeatVecDenseL2Simd<float>> encoder;
       

        // inline void save(FILE* fp) const {
        //     pecos::file_util::fput_multiple<uint32_t>(&num_codebooks, 1, fp);
        //     pecos::file_util::fput_multiple<int>(&dimension, 1, fp);
        //     pecos::file_util::fput_multiple<int>(&low_rank, 1, fp);
        //     pecos::file_util::fput_multiple<int>(&select, 1, fp);
        //     size_t sz = projection_matrix.size();
        //     pecos::file_util::fput_multiple<size_t>(&sz, 1, fp);
        //     if (sz) {
        //         pecos::file_util::fput_multiple<float>(&projection_matrix[0], sz, fp);
        //     }
        //     sz = codebook.size();
        //     pecos::file_util::fput_multiple<size_t>(&sz, 1, fp);
        //     if (sz) {
        //         pecos::file_util::fput_multiple<float>(&codebook[0], sz, fp);
        //     }
        // }

        // inline void load(FILE* fp) {
        //     pecos::file_util::fget_multiple<uint32_t>(&num_codebooks, 1, fp);
        //     pecos::file_util::fget_multiple<int>(&dimension, 1, fp);
        //     pecos::file_util::fget_multiple<int>(&low_rank, 1, fp);
        //     pecos::file_util::fget_multiple<int>(&select, 1, fp);
        //     size_t sz = 0;
        //     pecos::file_util::fget_multiple<size_t>(&sz, 1, fp);
        //     projection_matrix.resize(sz);
        //     if (sz) {
        //         pecos::file_util::fget_multiple<float>(&projection_matrix[0], sz, fp);
        //     }
        //     pecos::file_util::fget_multiple<size_t>(&sz, 1, fp);
        //     codebook.resize(sz);
        //     if (sz) {
        //         pecos::file_util::fget_multiple<float>(&codebook[0], sz, fp);
        //     }
        // }
        inline void setup() {
        // }
//         inline void compute_query_rplsh_code(uint64_t& result, const float* query_lowrank_projection_ptr) const {
// /*
//             result = 0;// =  (uint64_t) zzs[0] << 48 | (uint64_t) zzs[1] << 32 | (uint64_t) zzs[2] << 16 | zzs[3];
//             size_t offset = 48;
//             for (int i = 0; i < 4; i++) {
//                 __m512 _query_sub_vector = _mm512_loadu_ps(query_lowrank_projection_ptr);
//                 __m512i _qr =  _mm512_mask_blend_epi32(_mm512_cmp_ps_mask(_query_sub_vector, _falseValue, _CMP_LT_OQ), _trueValuei32, _falseValuei32);
//                 //zzs[i] = _mm512_movepi32_mask(_mm512_slli_epi32(_qr, 31));
//                 result += (uint64_t) _mm512_movepi32_mask(_mm512_slli_epi32(_qr, 31)) << offset;
//                 query_lowrank_projection_ptr += 16;
//                 offset -= 16;
//             }
// */
//             //uint32_t qq = query_rplsh_code;
// /*
//             __m512 _trueValue     = _mm512_set1_ps( 1.0f );
//             __m512 _falseValue    = _mm512_set1_ps( 0.0f );
//             std::cout<<std::endl;           
//             std::vector<float> result(32, 0);
//             compute_projection_information(query, result.data(), query_norm, query_squared_norm);
//             _mm512_storeu_ps(&result[0], _mm512_mask_blend_ps(_mm512_cmp_ps_mask(_mm512_loadu_ps(&result[0]), _mm512_set1_ps(0), _CMP_LT_OQ), _trueValue, _falseValue));
//             _mm512_storeu_ps(&result[16], _mm512_mask_blend_ps(_mm512_cmp_ps_mask(_mm512_loadu_ps(&result[16]), _mm512_set1_ps(0), _CMP_LT_OQ), _trueValue, _falseValue));
//             uint32_t tmp = 0; 
//             for (int i = 0 ; i < 32; i++) {

//                 //std::cout<<result[i]<<",";
//                 tmp += result[i];
//                 tmp <<= 1;
//             } 
// */
//             //std::cout<<std::endl;
//             /*   
//             std::vector<float> code_basis;
//             for (int i = 0 ; i < 32; i++) {
//                 code_basis.push_back(1 << i);
//             }
//             __m512 a = _mm512_mul_ps(_mm512_loadu_ps(&code_basis[0]), _mm512_loadu_ps(&result[0]));
//             __m512 b = _mm512_mul_ps(_mm512_loadu_ps(&code_basis[16]), _mm512_loadu_ps(&result[16]));
//             uint32_t aa = reinterpret_cast<uint32_t>(_mm512_reduce_add_ps(a));
//             uint32_t bb = reinterpret_cast<uint32_t>(_mm512_reduce_add_ps(b));
//             for (int i = 0 ; i < 16; i++) {
//                 std::cout<<(aa & 0b1)<<",";
//                 aa >>= 1;
//             }
//             std::cout<<std::endl; 
//             for (int i = 0 ; i < 16; i++) {
//                 std::cout<<(bb & 0b1)<<",";
//                 bb >>= 1;
//             }*/ 
//             //float bb = _mm512_reduce_add_ps(b);
//             //float r = _mm512_reduce_add_ps(_mm512_add_ps(a, b));
//             //float projected_qres_norm = _mm512_reduce_add_ps(_accumulator);
//             //std::cout<<r<<std::endl;
        }

        inline void compute_projection_information(const float* query, float* result, float& query_norm, float& query_squared_norm) const {
            // query_squared_norm = do_dot_product_simd(query, query, dimension); 
            query_squared_norm = finnerpfunc_(query, query, dist_func_param_);
            query_norm = std::sqrt(query_squared_norm);
            for (int i = 0; i < low_rank; i++) {
                // result[i] = do_dot_product_simd(query, &projection_matrix[i * dimension], dimension);
                result[i] = finnerpfunc_(query, &projection_matrix[i * dimension], dist_func_param_);
            } 
        }



        __attribute__((__target__("avx512f")))
        inline void compute_non_approximate_terms(const float* query, uint8_t* lut_ptr, float& scale, float& bias) const {
        }

        __attribute__((__target__("default")))
        inline void compute_non_approximate_terms(const float* query, uint8_t* lut_ptr, float& scale, float& bias) const {
        }

        __attribute__((__target__("avx512f")))
        inline void approximate_angular_distance(
            float* appx_result,
            const uint32_t& max_degree,
            const float& topk_ub_dist,
            const size_t neighbor_size,
            const float& query_norm,
            const float& query_squared_norm,
            const float* query_lowrank_projection, 
            const float& query_center_projection_coefficient,
            const char* stored_info,
            const float& ss=1,
            const float& bb=0
        ) const {
            // number of iterative loads
            size_t neighboring_float_size = sizeof(float) * 16;
            size_t neighboring_uint64_size = 64;
            size_t neighboring_index_size = sizeof(uint32_t) * 16;
            size_t center_size = 2 * sizeof(float);
            int rounds = neighbor_size % 16 == 0 ? neighbor_size / 16 : neighbor_size / 16 + 1;
            __m512 _trueValue     = _mm512_set1_ps( 1.0f );
            __m512 _falseValue    = _mm512_setzero_ps();
            __m512i _trueValuei32     = _mm512_set1_epi32( 1 );
            __m512i _falseValuei32    = _mm512_setzero_si512();
            float* appx_result_ptr = appx_result;
            
            stored_info += center_size;
            
            // process query information
            float qres_norm = std::sqrt ( 1 - query_center_projection_coefficient * query_center_projection_coefficient );
            // define returned mm512 values
            __m512 _topk_ub_dist = _mm512_set1_ps(topk_ub_dist);
            __m512 _query_center_projection_coefficient = _mm512_set1_ps(query_center_projection_coefficient);

            // compute normalized projected query residual vector

            __m512 _qres_norm = _mm512_set1_ps(qres_norm);
            
            const float* query_lowrank_projection_ptr = query_lowrank_projection;
            uint64_t talk2 = 0;// =  (uint64_t) zzs[0] << 48 | (uint64_t) zzs[1] << 32 | (uint64_t) zzs[2] << 16 | zzs[3];
            size_t offset = 48;
            //std::vector<uint16_t> zzs(4);
            for (int i = 0; i < 4; i++) {
                __m512 _query_sub_vector = _mm512_loadu_ps(query_lowrank_projection_ptr);
                __m512 _center_sub_vector = _mm512_loadu_ps(stored_info);
                _query_sub_vector = _mm512_fmsub_ps(_query_center_projection_coefficient, _center_sub_vector, _query_sub_vector);
                __m512i _qr =  _mm512_mask_blend_epi32(_mm512_cmp_ps_mask(_query_sub_vector, _mm512_setzero_ps(), _CMP_GE_OQ), _trueValuei32, _mm512_setzero_si512());
                talk2 += (uint64_t) _mm512_movepi32_mask(_mm512_slli_epi32(_qr, 31)) << offset;
                query_lowrank_projection_ptr += 16;
                offset -= 16;
                stored_info += neighboring_float_size;
            }
            __m512i _lookup_table = _mm512_set1_epi64(talk2);
            
            uint64_t talk3 = 0;// =  (uint64_t) zzs[0] << 48 | (uint64_t) zzs[1] << 32 | (uint64_t) zzs[2] << 16 | zzs[3];
            size_t offset2 = 48;
            //std::vector<uint16_t> zzs(4);
            for (int i = 0; i < 4; i++) {
                __m512 _query_sub_vector = _mm512_loadu_ps(query_lowrank_projection_ptr);
                __m512 _center_sub_vector = _mm512_loadu_ps(stored_info);
                _query_sub_vector = _mm512_fmsub_ps(_query_center_projection_coefficient, _center_sub_vector, _query_sub_vector);
                __m512i _qr =  _mm512_mask_blend_epi32(_mm512_cmp_ps_mask(_query_sub_vector, _mm512_setzero_ps(), _CMP_GE_OQ), _trueValuei32, _mm512_setzero_si512());
                talk3 += (uint64_t) _mm512_movepi32_mask(_mm512_slli_epi32(_qr, 31)) << offset2;
                query_lowrank_projection_ptr += 16;
                offset2 -= 16;
                stored_info += neighboring_float_size;
            }
            __m512i _lookup_table2 = _mm512_set1_epi64(talk3);


            __m512i _mask = _mm512_set1_epi8(0x0f);  
            __m512i _mask00ff = _mm512_set1_epi16(0x00ff);  
            __m512i _mask0000ffff = _mm512_set1_epi32(0x0000ffff);  
            __m512i _mask00000000ffffffff = _mm512_set1_epi64(0x00000000ffffffff);  

            for (int i = 0; i < rounds; i++) {
                // compute |dres|^2 
                __m512 _neighbor_res_norm = _mm512_loadu_ps(stored_info);
                stored_info += neighboring_float_size;
                // compute |qproj - dproj|^2
                __m512 _neighbor_center_projection_coefficient = _mm512_loadu_ps(stored_info);
                stored_info += neighboring_float_size;

                __m512i _points = _mm512_loadu_si512((__m512i const*)stored_info);
                stored_info += neighboring_uint64_size;
                __m512i _xor = _mm512_xor_si512(_points, _lookup_table);

                __m512i _s = _mm512_setzero_si512();
                __m512i _low = _mm512_and_si512(_xor, _mask);
                __m512i _high = _mm512_and_si512(_mm512_srli_epi16(_xor, 4), _mask);
                __m512i _pl = _mm512_shuffle_epi8(_popcnt_lookup_table, _low);
                __m512i _ph = _mm512_shuffle_epi8(_popcnt_lookup_table, _high);
                _s = _mm512_add_epi8(_s, _pl);
                _s = _mm512_add_epi8(_s, _ph);

                _s = _mm512_add_epi16(_mm512_and_si512(_s, _mask00ff), _mm512_and_si512(_mm512_srli_epi16(_s, 8), _mask00ff));
                _s = _mm512_add_epi32(_mm512_and_si512(_s, _mask0000ffff), _mm512_and_si512(_mm512_srli_epi32(_s, 16), _mask0000ffff));
                _s = _mm512_add_epi64(_mm512_and_si512(_s, _mask00000000ffffffff), _mm512_and_si512(_mm512_srli_epi64(_s, 32), _mask00000000ffffffff));
                __m256i _s1 = _mm512_cvtepi64_epi32(_s);

                _points = _mm512_loadu_si512((__m512i const*)stored_info);
                stored_info += neighboring_uint64_size;
                _xor = _mm512_xor_si512(_points, _lookup_table);

                _s = _mm512_setzero_si512(); //mm512_set1_epi32( 0 );//_falseValuei32;//_mm512_setzero_si512();
                _low = _mm512_and_si512(_xor, _mask);
                _high = _mm512_and_si512(_mm512_srli_epi16(_xor, 4), _mask);
                _pl = _mm512_shuffle_epi8(_popcnt_lookup_table, _low);
                _ph = _mm512_shuffle_epi8(_popcnt_lookup_table, _high);
                _s = _mm512_add_epi8(_s, _pl);
                _s = _mm512_add_epi8(_s, _ph);

                _s = _mm512_add_epi16(_mm512_and_si512(_s, _mask00ff), _mm512_and_si512(_mm512_srli_epi16(_s, 8), _mask00ff));
                _s = _mm512_add_epi32(_mm512_and_si512(_s, _mask0000ffff), _mm512_and_si512(_mm512_srli_epi32(_s, 16), _mask0000ffff));
                _s = _mm512_add_epi64(_mm512_and_si512(_s, _mask00000000ffffffff), _mm512_and_si512(_mm512_srli_epi64(_s, 32), _mask00000000ffffffff));
                __m256i _s2 = _mm512_cvtepi64_epi32(_s);

                __m512i _s_total = _mm512_castsi256_si512(_s1);
                _s_total = _mm512_inserti64x4(_s_total, _s2, 1); 


                _points = _mm512_loadu_si512((__m512i const*)stored_info);
                stored_info += neighboring_uint64_size;
                _xor = _mm512_xor_si512(_points, _lookup_table2);
                _s = _mm512_setzero_si512(); //mm512_set1_epi32( 0 );//_falseValuei32;//_mm512_setzero_si512();
                _low = _mm512_and_si512(_xor, _mask);
                _high = _mm512_and_si512(_mm512_srli_epi16(_xor, 4), _mask);
                _pl = _mm512_shuffle_epi8(_popcnt_lookup_table, _low);
                _ph = _mm512_shuffle_epi8(_popcnt_lookup_table, _high);
                _s = _mm512_add_epi8(_s, _pl);
                _s = _mm512_add_epi8(_s, _ph);

                _s = _mm512_add_epi16(_mm512_and_si512(_s, _mask00ff), _mm512_and_si512(_mm512_srli_epi16(_s, 8), _mask00ff));
                _s = _mm512_add_epi32(_mm512_and_si512(_s, _mask0000ffff), _mm512_and_si512(_mm512_srli_epi32(_s, 16), _mask0000ffff));
                _s = _mm512_add_epi64(_mm512_and_si512(_s, _mask00000000ffffffff), _mm512_and_si512(_mm512_srli_epi64(_s, 32), _mask00000000ffffffff));
                __m256i _s3 = _mm512_cvtepi64_epi32(_s);

                _points = _mm512_loadu_si512((__m512i const*)stored_info);
                stored_info += neighboring_uint64_size;
                _xor = _mm512_xor_si512(_points, _lookup_table2);

                _s = _mm512_setzero_si512(); //mm512_set1_epi32( 0 );//_falseValuei32;//_mm512_setzero_si512();
                _low = _mm512_and_si512(_xor, _mask);
                _high = _mm512_and_si512(_mm512_srli_epi16(_xor, 4), _mask);
                _pl = _mm512_shuffle_epi8(_popcnt_lookup_table, _low);
                _ph = _mm512_shuffle_epi8(_popcnt_lookup_table, _high);
                _s = _mm512_add_epi8(_s, _pl);
                _s = _mm512_add_epi8(_s, _ph);

                _s = _mm512_add_epi16(_mm512_and_si512(_s, _mask00ff), _mm512_and_si512(_mm512_srli_epi16(_s, 8), _mask00ff));
                _s = _mm512_add_epi32(_mm512_and_si512(_s, _mask0000ffff), _mm512_and_si512(_mm512_srli_epi32(_s, 16), _mask0000ffff));
                _s = _mm512_add_epi64(_mm512_and_si512(_s, _mask00000000ffffffff), _mm512_and_si512(_mm512_srli_epi64(_s, 32), _mask00000000ffffffff));
                __m256i _s4 = _mm512_cvtepi64_epi32(_s);

                __m512i _s_total2 = _mm512_castsi256_si512(_s3);
                _s_total2 = _mm512_inserti64x4(_s_total2, _s4, 1); 

                _s_total = _mm512_add_epi32(_s_total, _s_total2);
                //std::vector<uint32_t> h2(16);
                //_mm512_storeu_si512(&h2[0], _s_total);
                //for(int i = 0 ;i < 16;i++) {
                //    std::cout<<h2[i]<<",";
                //} std::cout<<std::endl;
                //_s_total = _mm512_inserti64x4(_s_total, _s2, 1); 
                //__m512i _s_max = _mm512_set1_epi32(ultimate_select + 16);
                //__m512i _s_min = _mm512_set1_epi32(ultimate_select - 16);

                //_s_total = _mm512_mask_blend_epi32(_mm512_cmp_epi32_mask(_s_total, _s_max , _MM_CMPINT_LE), _s_max, _s_total);
                //_s_total = _mm512_mask_blend_epi32(_mm512_cmp_epi32_mask(_s_total, _s_min , _MM_CMPINT_LE), _s_total, _s_min);



                //__m512 _tmp1 = _mm512_permutexvar_ps(_s_total, _cos_table0);
                //__m512 _tmp2 = _mm512_permutexvar_ps(_s_total, _cos_table1);
                __m512 _tmp3 = _mm512_permutexvar_ps(_s_total, _cos_table2);
                __m512 _tmp4 = _mm512_permutexvar_ps(_s_total, _cos_table3);
                __m512 _tmp5 = _mm512_permutexvar_ps(_s_total, _cos_table4);
                __m512 _tmp6 = _mm512_permutexvar_ps(_s_total, _cos_table5);
                //__m512 _tmp7 = _mm512_permutexvar_ps(_s_total, _cos_table6);
                //__m512 _tmp8 = _mm512_permutexvar_ps(_s_total, _cos_table7);
                __m512 _qres_dres_cos_value = _mm512_mask_blend_ps(_mm512_cmp_epi32_mask(_s_total, _mm512_set1_epi32(79), _MM_CMPINT_LE), _tmp6, _tmp5);
                //_qres_dres_cos_value = _mm512_mask_blend_ps(_mm512_cmp_epi32_mask(_s_total, _mm512_set1_epi32(95), _MM_CMPINT_LE), _qres_dres_cos_value, _tmp6);
                //_qres_dres_cos_value = _mm512_mask_blend_ps(_mm512_cmp_epi32_mask(_s_total, _mm512_set1_epi32(79), _MM_CMPINT_LE), _qres_dres_cos_value, _tmp5);
                _qres_dres_cos_value = _mm512_mask_blend_ps(_mm512_cmp_epi32_mask(_s_total, _mm512_set1_epi32(63), _MM_CMPINT_LE), _qres_dres_cos_value, _tmp4);
                _qres_dres_cos_value = _mm512_mask_blend_ps(_mm512_cmp_epi32_mask(_s_total, _mm512_set1_epi32(47), _MM_CMPINT_LE), _qres_dres_cos_value, _tmp3);
                //_qres_dres_cos_value = _mm512_mask_blend_ps(_mm512_cmp_epi32_mask(_s_total, _mm512_set1_epi32(31), _MM_CMPINT_LE), _qres_dres_cos_value, _tmp2);
                //_qres_dres_cos_value = _mm512_mask_blend_ps(_mm512_cmp_epi32_mask(_s_total, _mm512_set1_epi32(15), _MM_CMPINT_LE), _qres_dres_cos_value, _tmp1);



                //std::vector<float> ff(16);
                //_mm512_storeu_ps(ff.data(), _qres_dres_cos_value);
                //for(int j = 0; j < 16;j++) { std::cout<<ff[j]<<",";} std::cout<<std::endl;


                //__m512 _neighbor_res_squared_norm = _mm512_mul_ps(_neighbor_res_norm, _neighbor_res_norm);
                __m512 _qproj_dproj_ip = _mm512_mul_ps(_query_center_projection_coefficient, _neighbor_center_projection_coefficient);
               
                //_qres_dres_cos_value = _mm512_fmadd_ps(_correct_scale, _qres_dres_cos_value, _correct_bias);
                //__m512 _qres_dres_ip = _mm512_mul_ps(_qres_dres_cos_value, _mm512_mul_ps(_qres_norm, _neighbor_res_norm));
                __m512 _qres_dres_ip = _mm512_mul_ps(_qres_dres_cos_value, _mm512_mul_ps(_qres_norm, _neighbor_res_norm));
                //__m512 _qres_dres_ip = _mm512_mul_ps(_qres_dres_cos_value, _neighbor_res_norm);
                __m512 _appx_ip_dist = _mm512_sub_ps(_trueValue, _mm512_add_ps(_qres_dres_ip, _qproj_dproj_ip));
                //_mm512_storeu_ps(&appx_result_ptr[0], _appx_ip_dist);
                //for (int r = 0; r <=  15; r++) { std::cout<<appx_result_ptr[r]<<",";  } std::cout<<std::endl;
                _mm512_storeu_ps(&appx_result_ptr[0], _mm512_mask_blend_ps(_mm512_cmp_ps_mask(_appx_ip_dist, _topk_ub_dist, _CMP_LT_OQ), _falseValue, _trueValue));

                appx_result_ptr += 16;

            }
        }
 
        __attribute__((__target__("avx512f")))
        inline void approximate_ip_distance(
            float* appx_result,
            const uint32_t& max_degree,
            const float& topk_ub_dist,
            const size_t neighbor_size,
            const float& query_norm,
            const float& query_squared_norm,
            const float* query_lowrank_projection, 
            const float& center_query_ip,
            const char* stored_info,
            const float& ss=1,
            const float& bb=0
        ) const {
            // number of iterative loads
            size_t neighboring_float_size = sizeof(float) * 16;
            size_t neighboring_uint64_size = 64;
            //size_t neighboring_residual_vector_size = sizeof(float) * low_rank;
            size_t neighboring_index_size = sizeof(uint32_t) * 16;
            size_t center_size = 2 * sizeof(float);
            int rounds = neighbor_size % 16 == 0 ? neighbor_size / 16 : neighbor_size / 16 + 1;
            __m512 _correct_bias = _mm512_set1_ps(bb);
            __m512 _correct_scale = _mm512_set1_ps(ss);
            //int num_left_index = rounds * 16 == neighbor_size ? 0 : neighbor_size - rounds * 16;
            float* appx_result_ptr = appx_result;
             
            float center_node_norm = reinterpret_cast<const float*>(stored_info)[0];
            float center_node_squared_norm = reinterpret_cast<const float*>(stored_info)[1]; 
            
            stored_info += center_size;
            //for (int r = 8; r < 16; r++) { 
            //    _mm_prefetch(stored_info + r * 64, _MM_HINT_T0);
            //} 
            // process query information
            float cos_value = center_query_ip / query_norm / center_node_norm;
            float sin_value = std::sqrt ( 1 - cos_value * cos_value );
            float qres_norm = query_norm * sin_value;
            float query_center_projection_coefficient = center_query_ip / center_node_squared_norm;
            
            // define returned mm512 values
            __m512 _trueValue     = _mm512_set1_ps( 1.0f );
            __m512 _falseValue    = _mm512_set1_ps( 0.0f );
            __m512i _trueValuei32     = _mm512_set1_epi32( 1 );
            __m512i _falseValuei32    = _mm512_setzero_si512();
            __m512 _topk_ub_dist = _mm512_set1_ps(topk_ub_dist);
            //std::cout<<"query center l2 : "<<center_query_l2_distance<<" COS : "<<cos_value<<" SIN : "<<sin_value<<" QUERY^2 : "<<query_squared_norm<<" CENTER^2 : "<<center_node_squared_norm<<" QUERY^T CENTER : "<<query_center_ip<<std::endl;
            // compute |qproj - dproj|^2 + |qres|^2 + |dres|^2 + Qres Dres appx IP
            __m512 _center_node_squared_norm = _mm512_set1_ps(center_node_squared_norm);
            __m512 _query_center_projection_coefficient = _mm512_set1_ps(query_center_projection_coefficient);
            __m512 _qres_norm = _mm512_set1_ps(qres_norm);
            

            // compute normalized projected query residual vector
            const float* query_lowrank_projection_ptr = query_lowrank_projection;
            std::vector<uint16_t> zzs(4);
            for (int i = 0; i < 4; i++) {
                __m512 _query_sub_vector = _mm512_loadu_ps(query_lowrank_projection_ptr);
                __m512 _center_sub_vector = _mm512_loadu_ps(stored_info);
                _query_sub_vector = _mm512_sub_ps(_query_sub_vector, _mm512_mul_ps(_query_center_projection_coefficient, _center_sub_vector));
                __m512i _qr =  _mm512_mask_blend_epi32(_mm512_cmp_ps_mask(_query_sub_vector, _falseValue, _CMP_LT_OQ), _trueValuei32, _falseValuei32);
                //__m512i _qr =  _mm512_mask_blend_epi32(_mm512_cmp_ps_mask(_query_sub_vector, _falseValue, _CMP_LT_OQ), _mm512_set1_epi32(1), _mm512_setzero_si512());
                zzs[i] = _mm512_movepi32_mask(_mm512_slli_epi32(_qr, 31));
                query_lowrank_projection_ptr += 16;
                stored_info += neighboring_float_size;
            }
            //_mm_prefetch(stored_info, _MM_HINT_T0);

            //const auto codebook_index_ptr = reinterpret_cast<const uint32_t*>(stored_info);     
            //float projected_qres_norm = _mm512_reduce_add_ps(_accumulator);
            //projected_qres_norm = std::sqrt(projected_qres_norm); 
            //__m512 _projected_qres_norm = _mm512_set1_ps(projected_qres_norm);

            uint64_t talk2 =  (uint64_t) zzs[0] << 48 | (uint64_t) zzs[1] << 32 | (uint64_t) zzs[2] << 16 | zzs[3];
            __m512i _lookup_table = _mm512_set1_epi64(talk2);

            __m512i _mask = _mm512_set1_epi8(0x0f);  
            __m512i _mask00ff = _mm512_set1_epi16(0x00ff);  
            __m512i _mask0000ffff = _mm512_set1_epi32(0x0000ffff);  
            __m512i _mask00000000ffffffff = _mm512_set1_epi64(0x00000000ffffffff);  

            for (int i = 0; i < rounds; i++) {
                // compute |dres|^2 
                __m512 _neighbor_res_norm = _mm512_loadu_ps(stored_info);
                stored_info += neighboring_float_size;
                // compute |qproj - dproj|^2
                __m512 _neighbor_center_projection_coefficient = _mm512_loadu_ps(stored_info);
                stored_info += neighboring_float_size;

                //__m512 _neighbor_res_squared_norm = _mm512_mul_ps(_neighbor_res_norm, _neighbor_res_norm);
                __m512 _qproj_dproj_ip = _mm512_mul_ps(_mm512_mul_ps(_query_center_projection_coefficient, _neighbor_center_projection_coefficient), _center_node_squared_norm);

                __m512i _points = _mm512_loadu_si512((__m512i const*)stored_info);
                stored_info += neighboring_uint64_size;
                __m512i _xor = _mm512_xor_si512(_points, _lookup_table);

                //  lookup table trick
                __m512i _s = _falseValuei32;//_mm512_setzero_si512();
                __m512i _low = _mm512_and_si512(_xor, _mask);
                __m512i _high = _mm512_and_si512(_mm512_srli_epi16(_xor, 4), _mask);
                __m512i _pl = _mm512_shuffle_epi8(_popcnt_lookup_table, _low);
                __m512i _ph = _mm512_shuffle_epi8(_popcnt_lookup_table, _high);
                _s = _mm512_add_epi8(_s, _pl);
                _s = _mm512_add_epi8(_s, _ph);

                _s = _mm512_add_epi16(_mm512_and_si512(_s, _mask00ff), _mm512_and_si512(_mm512_srli_epi16(_s, 8), _mask00ff));
                _s = _mm512_add_epi32(_mm512_and_si512(_s, _mask0000ffff), _mm512_and_si512(_mm512_srli_epi32(_s, 16), _mask0000ffff));
                _s = _mm512_add_epi64(_mm512_and_si512(_s, _mask00000000ffffffff), _mm512_and_si512(_mm512_srli_epi64(_s, 32), _mask00000000ffffffff));
                __m256i _s1 = _mm512_cvtepi64_epi32(_s);

                _points = _mm512_loadu_si512((__m512i const*)stored_info);
                stored_info += neighboring_uint64_size;
                _xor = _mm512_xor_si512(_points, _lookup_table);

                //  lookup table trick
                _s = _falseValuei32;//_mm512_setzero_si512();
                _low = _mm512_and_si512(_xor, _mask);
                _high = _mm512_and_si512(_mm512_srli_epi16(_xor, 4), _mask);
                _pl = _mm512_shuffle_epi8(_popcnt_lookup_table, _low);
                _ph = _mm512_shuffle_epi8(_popcnt_lookup_table, _high);
                _s = _mm512_add_epi8(_s, _pl);
                _s = _mm512_add_epi8(_s, _ph);

                _s = _mm512_add_epi16(_mm512_and_si512(_s, _mask00ff), _mm512_and_si512(_mm512_srli_epi16(_s, 8), _mask00ff));
                _s = _mm512_add_epi32(_mm512_and_si512(_s, _mask0000ffff), _mm512_and_si512(_mm512_srli_epi32(_s, 16), _mask0000ffff));
                _s = _mm512_add_epi64(_mm512_and_si512(_s, _mask00000000ffffffff), _mm512_and_si512(_mm512_srli_epi64(_s, 32), _mask00000000ffffffff));
                __m256i _s2 = _mm512_cvtepi64_epi32(_s);

                __m512i _s_total = _mm512_castsi256_si512(_s1);
                _s_total = _mm512_inserti64x4(_s_total, _s2, 1); 
                //__m512i _s_max = _mm512_set1_epi32(48);
                //__m512i _s_min = _mm512_set1_epi32(16);
                __m512i _s_max = _mm512_set1_epi32(ultimate_select + 16);
                __m512i _s_min = _mm512_set1_epi32(ultimate_select - 16);

                _s_total = _mm512_mask_blend_epi32(_mm512_cmp_epi32_mask(_s_total, _s_max , _MM_CMPINT_LE), _s_max, _s_total);
                _s_total = _mm512_mask_blend_epi32(_mm512_cmp_epi32_mask(_s_total, _s_min , _MM_CMPINT_LE), _s_total, _s_min);
                __m512 _tmp1 = _mm512_permutexvar_ps(_s_total, _cos_table0);
                __m512 _tmp2 = _mm512_permutexvar_ps(_s_total, _cos_table1);
                __m512 _qres_dres_cos_value = _mm512_mask_blend_ps(_mm512_cmp_epi32_mask(_s_total, _cos_select, _MM_CMPINT_LE), _tmp1, _tmp2);

                _qres_dres_cos_value = _mm512_fmadd_ps(_correct_scale, _qres_dres_cos_value, _correct_bias);
                __m512 _qres_dres_ip = _mm512_mul_ps(_qres_dres_cos_value, _mm512_mul_ps(_qres_norm, _neighbor_res_norm));
                __m512 _appx_ip_dist = _mm512_sub_ps(_trueValue, _mm512_add_ps(_qres_dres_ip, _qproj_dproj_ip));
                //_mm512_storeu_ps(&appx_result_ptr[0], _appx_ip_dist);
                //for (int r = 0; r <=  15; r++) { std::cout<<appx_result_ptr[r]<<",";  } std::cout<<std::endl;
                _mm512_storeu_ps(&appx_result_ptr[0], _mm512_mask_blend_ps(_mm512_cmp_ps_mask(_appx_ip_dist, _topk_ub_dist, _CMP_LT_OQ), _falseValue, _trueValue));

                //stored_info += neighboring_index_size;
                appx_result_ptr += 16;

            }
        }

        __attribute__((__target__("avx512f")))
        inline void approximate_distance(
            float* appx_result,
            const uint32_t& max_degree,
            const float& topk_ub_dist,
            const size_t neighbor_size,
            const float& query_norm,
            const float& query_squared_norm,
            const float* query_lowrank_projection, 
            const float& center_query_l2_distance,
            const char* stored_info,
            const float& ss=1,
            const float& bb=0
        ) const {
            // number of iterative loads
            size_t neighboring_float_size = sizeof(float) * 16;
            size_t neighboring_uint64_size = 64;
            //size_t neighboring_residual_vector_size = sizeof(float) * low_rank;
            size_t neighboring_index_size = sizeof(uint32_t) * 16;
            size_t center_size = 2 * sizeof(float);
            int num_dimension_blocks = low_rank / 16;
            int rounds = neighbor_size % 16 == 0 ? neighbor_size / 16 : neighbor_size / 16 + 1;
            //__m512 _correct_bias = _mm512_set1_ps(bb);
            //__m512 _correct_scale = _mm512_set1_ps(ss);
            //int num_left_index = rounds * 16 == neighbor_size ? 0 : neighbor_size - rounds * 16;
            float* appx_result_ptr = appx_result;
            float center_node_norm = reinterpret_cast<const float*>(stored_info)[0];

            // std::cout<<" "<<center_node_norm<<"\n";
            float center_node_squared_norm = reinterpret_cast<const float*>(stored_info)[1]; 
            stored_info += center_size;
            //for (int r = 0; r < (neighbor_size  + 1) * num_dimension_blocks + rounds * 2; r++) { 
            //    _mm_prefetch(stored_info + r * 64, _MM_HINT_T0);
            //} 
            // process query information
            float query_center_ip = (center_node_squared_norm + query_squared_norm - center_query_l2_distance) * 0.5;
            float cos_value = query_center_ip / query_norm / center_node_norm;
            float sin_value = std::sqrt ( 1 - cos_value * cos_value );
            float qres_norm = query_norm * sin_value;
            float qres_squared_norm = qres_norm * qres_norm;
            float query_center_projection_coefficient = query_center_ip / center_node_squared_norm;
            // define returned mm512 values
            __m512 _minus2 = _mm512_set1_ps(-2);
            __m512 _trueValue     = _mm512_set1_ps( 1.0f );
            __m512 _falseValue    = _mm512_setzero_ps( );
            __m512i _trueValuei32     = _mm512_set1_epi32( 1 );
            __m512i _falseValuei32    = _mm512_setzero_si512( );
            __m512 _topk_ub_dist = _mm512_set1_ps(topk_ub_dist);
            //std::cout<<"query center l2 : "<<center_query_l2_distance<<" COS : "<<cos_value<<" SIN : "<<sin_value<<" QUERY^2 : "<<query_squared_norm<<" CENTER^2 : "<<center_node_squared_norm<<" QUERY^T CENTER : "<<query_center_ip<<std::endl;
            // compute |qproj - dproj|^2 + |qres|^2 + |dres|^2 + Qres Dres appx IP
            __m512 _center_node_squared_norm = _mm512_set1_ps(center_node_squared_norm);
            __m512 _query_center_projection_coefficient = _mm512_set1_ps(query_center_projection_coefficient);
            __m512 _qres_squared_norm = _mm512_set1_ps(qres_squared_norm);
            __m512 _qres_norm = _mm512_set1_ps(qres_norm);
            



            const float* query_lowrank_projection_ptr = query_lowrank_projection;
            uint64_t talk2 = 0;// =  (uint64_t) zzs[0] << 48 | (uint64_t) zzs[1] << 32 | (uint64_t) zzs[2] << 16 | zzs[3];
            size_t offset = 48;
            //std::vector<uint16_t> zzs(4);
            for (int i = 0; i < 4; i++) {
                __m512 _query_sub_vector = _mm512_loadu_ps(query_lowrank_projection_ptr);
                __m512 _center_sub_vector = _mm512_loadu_ps(stored_info);
                _query_sub_vector = _mm512_fmsub_ps(_query_center_projection_coefficient, _center_sub_vector, _query_sub_vector);
                __m512i _qr =  _mm512_mask_blend_epi32(_mm512_cmp_ps_mask(_query_sub_vector, _mm512_setzero_ps(), _CMP_GE_OQ), _trueValuei32, _mm512_setzero_si512());
                talk2 += (uint64_t) _mm512_movepi32_mask(_mm512_slli_epi32(_qr, 31)) << offset;
                query_lowrank_projection_ptr += 16;
                offset -= 16;
                stored_info += neighboring_float_size;
            }
            __m512i _lookup_table = _mm512_set1_epi64(talk2);
            
            uint64_t talk3 = 0;// =  (uint64_t) zzs[0] << 48 | (uint64_t) zzs[1] << 32 | (uint64_t) zzs[2] << 16 | zzs[3];
            size_t offset2 = 48;
            //std::vector<uint16_t> zzs(4);
            for (int i = 0; i < 4; i++) {
                __m512 _query_sub_vector = _mm512_loadu_ps(query_lowrank_projection_ptr);
                __m512 _center_sub_vector = _mm512_loadu_ps(stored_info);
                _query_sub_vector = _mm512_fmsub_ps(_query_center_projection_coefficient, _center_sub_vector, _query_sub_vector);
                __m512i _qr =  _mm512_mask_blend_epi32(_mm512_cmp_ps_mask(_query_sub_vector, _mm512_setzero_ps(), _CMP_GE_OQ), _trueValuei32, _mm512_setzero_si512());
                talk3 += (uint64_t) _mm512_movepi32_mask(_mm512_slli_epi32(_qr, 31)) << offset2;
                query_lowrank_projection_ptr += 16;
                offset2 -= 16;
                stored_info += neighboring_float_size;
            }
            __m512i _lookup_table2 = _mm512_set1_epi64(talk3);

            for (int i = 0; i < rounds; i++) {
                // compute |dres|^2 
                __m512 _neighbor_res_norm = _mm512_loadu_ps(stored_info);
                stored_info += neighboring_float_size;
                // compute |qproj - dproj|^2
                __m512 _neighbor_center_projection_coefficient = _mm512_loadu_ps(stored_info);
                stored_info += neighboring_float_size;

                __m512 _neighbor_res_squared_norm_plus_qres_squared_norm = _mm512_fmadd_ps(_neighbor_res_norm, _neighbor_res_norm, _qres_squared_norm);
                __m512 _qproj_dproj_diff = _mm512_sub_ps(_query_center_projection_coefficient, _neighbor_center_projection_coefficient);
                __m512 _exact_l2_distance = _mm512_fmadd_ps(_mm512_mul_ps(_qproj_dproj_diff, _qproj_dproj_diff), _center_node_squared_norm, _neighbor_res_squared_norm_plus_qres_squared_norm);
                // combine exact distances
                //__m512 _exact_l2_distance = _qproj_dproj_l2 + _qres_squared_norm + _neighbor_res_squared_norm;
                //std::vector<float> ff(16);
                //_mm512_storeu_ps(ff.data(), _exact_l2_distance);
                //for(int j = 0; j < 16; j++) { std::cout<<ff[j]<<",";} std::cout<<std::endl; 
             
                __m512i _mask = _mm512_set1_epi8(0x0f);  
                __m512i _mask00ff = _mm512_set1_epi16(0x00ff);  
                __m512i _mask0000ffff = _mm512_set1_epi32(0x0000ffff);  
                __m512i _mask00000000ffffffff = _mm512_set1_epi64(0x00000000ffffffff);  

                __m512i _points = _mm512_loadu_si512((__m512i const*)stored_info);
                stored_info += neighboring_uint64_size;
                __m512i _xor = _mm512_xor_si512(_points, _lookup_table);

                __m512i _s = _mm512_setzero_si512();
                __m512i _low = _mm512_and_si512(_xor, _mask);
                __m512i _high = _mm512_and_si512(_mm512_srli_epi16(_xor, 4), _mask);
                __m512i _pl = _mm512_shuffle_epi8(_popcnt_lookup_table, _low);
                __m512i _ph = _mm512_shuffle_epi8(_popcnt_lookup_table, _high);
                _s = _mm512_add_epi8(_s, _pl);
                _s = _mm512_add_epi8(_s, _ph);

                _s = _mm512_add_epi16(_mm512_and_si512(_s, _mask00ff), _mm512_and_si512(_mm512_srli_epi16(_s, 8), _mask00ff));
                _s = _mm512_add_epi32(_mm512_and_si512(_s, _mask0000ffff), _mm512_and_si512(_mm512_srli_epi32(_s, 16), _mask0000ffff));
                _s = _mm512_add_epi64(_mm512_and_si512(_s, _mask00000000ffffffff), _mm512_and_si512(_mm512_srli_epi64(_s, 32), _mask00000000ffffffff));
                __m256i _s1 = _mm512_cvtepi64_epi32(_s);

                // std::vector<uint64_t> h(16);
                // _mm512_storeu_si512(h.data(), _points);
                // for(int j = 0; j < 8; j++) { std::cout<<h[j]<<",";}

                

                _points = _mm512_loadu_si512((__m512i const*)stored_info);
                stored_info += neighboring_uint64_size;
                _xor = _mm512_xor_si512(_points, _lookup_table);

                _s = _mm512_setzero_si512(); //mm512_set1_epi32( 0 );//_falseValuei32;//_mm512_setzero_si512();
                _low = _mm512_and_si512(_xor, _mask);
                _high = _mm512_and_si512(_mm512_srli_epi16(_xor, 4), _mask);
                _pl = _mm512_shuffle_epi8(_popcnt_lookup_table, _low);
                _ph = _mm512_shuffle_epi8(_popcnt_lookup_table, _high);
                _s = _mm512_add_epi8(_s, _pl);
                _s = _mm512_add_epi8(_s, _ph);

                _s = _mm512_add_epi16(_mm512_and_si512(_s, _mask00ff), _mm512_and_si512(_mm512_srli_epi16(_s, 8), _mask00ff));
                _s = _mm512_add_epi32(_mm512_and_si512(_s, _mask0000ffff), _mm512_and_si512(_mm512_srli_epi32(_s, 16), _mask0000ffff));
                _s = _mm512_add_epi64(_mm512_and_si512(_s, _mask00000000ffffffff), _mm512_and_si512(_mm512_srli_epi64(_s, 32), _mask00000000ffffffff));
                __m256i _s2 = _mm512_cvtepi64_epi32(_s);

                __m512i _s_total = _mm512_castsi256_si512(_s1);
                _s_total = _mm512_inserti64x4(_s_total, _s2, 1); 


                _points = _mm512_loadu_si512((__m512i const*)stored_info);
                stored_info += neighboring_uint64_size;
                _xor = _mm512_xor_si512(_points, _lookup_table2);
                _s = _mm512_setzero_si512(); //mm512_set1_epi32( 0 );//_falseValuei32;//_mm512_setzero_si512();
                _low = _mm512_and_si512(_xor, _mask);
                _high = _mm512_and_si512(_mm512_srli_epi16(_xor, 4), _mask);
                _pl = _mm512_shuffle_epi8(_popcnt_lookup_table, _low);
                _ph = _mm512_shuffle_epi8(_popcnt_lookup_table, _high);
                _s = _mm512_add_epi8(_s, _pl);
                _s = _mm512_add_epi8(_s, _ph);

                _s = _mm512_add_epi16(_mm512_and_si512(_s, _mask00ff), _mm512_and_si512(_mm512_srli_epi16(_s, 8), _mask00ff));
                _s = _mm512_add_epi32(_mm512_and_si512(_s, _mask0000ffff), _mm512_and_si512(_mm512_srli_epi32(_s, 16), _mask0000ffff));
                _s = _mm512_add_epi64(_mm512_and_si512(_s, _mask00000000ffffffff), _mm512_and_si512(_mm512_srli_epi64(_s, 32), _mask00000000ffffffff));
                __m256i _s3 = _mm512_cvtepi64_epi32(_s);

                _points = _mm512_loadu_si512((__m512i const*)stored_info);
                stored_info += neighboring_uint64_size;
                _xor = _mm512_xor_si512(_points, _lookup_table2);

                _s = _mm512_setzero_si512(); //mm512_set1_epi32( 0 );//_falseValuei32;//_mm512_setzero_si512();
                _low = _mm512_and_si512(_xor, _mask);
                _high = _mm512_and_si512(_mm512_srli_epi16(_xor, 4), _mask);
                _pl = _mm512_shuffle_epi8(_popcnt_lookup_table, _low);
                _ph = _mm512_shuffle_epi8(_popcnt_lookup_table, _high);
                _s = _mm512_add_epi8(_s, _pl);
                _s = _mm512_add_epi8(_s, _ph);

                _s = _mm512_add_epi16(_mm512_and_si512(_s, _mask00ff), _mm512_and_si512(_mm512_srli_epi16(_s, 8), _mask00ff));
                _s = _mm512_add_epi32(_mm512_and_si512(_s, _mask0000ffff), _mm512_and_si512(_mm512_srli_epi32(_s, 16), _mask0000ffff));
                _s = _mm512_add_epi64(_mm512_and_si512(_s, _mask00000000ffffffff), _mm512_and_si512(_mm512_srli_epi64(_s, 32), _mask00000000ffffffff));
                __m256i _s4 = _mm512_cvtepi64_epi32(_s);

                __m512i _s_total2 = _mm512_castsi256_si512(_s3);
                _s_total2 = _mm512_inserti64x4(_s_total2, _s4, 1); 

                _s_total = _mm512_add_epi32(_s_total, _s_total2);
                //__m512i _s_max = _mm512_set1_epi32(80);
                //__m512i _s_min = _mm512_set1_epi32(48);


                
                // std::vector<uint32_t> h(16);
                // _mm512_storeu_si512(h.data(), _s_total2);
                // for(int j = 0; j < 16; j++) { std::cout<<h[j]<<",";}


                //__m512 _tmp1 = _mm512_permutexvar_ps(_s_total, _cos_table0);
                //__m512 _tmp2 = _mm512_permutexvar_ps(_s_total, _cos_table1);
                __m512 _tmp3 = _mm512_permutexvar_ps(_s_total, _cos_table2);
                __m512 _tmp4 = _mm512_permutexvar_ps(_s_total, _cos_table3);
                __m512 _tmp5 = _mm512_permutexvar_ps(_s_total, _cos_table4);
                __m512 _tmp6 = _mm512_permutexvar_ps(_s_total, _cos_table5);
                //__m512 _tmp7 = _mm512_permutexvar_ps(_s_total, _cos_table6);
                //__m512 _tmp8 = _mm512_permutexvar_ps(_s_total, _cos_table7);
                __m512 _qres_dres_cos_value = _mm512_mask_blend_ps(_mm512_cmp_epi32_mask(_s_total, _mm512_set1_epi32(79), _MM_CMPINT_LE), _tmp6, _tmp5);
                //_qres_dres_cos_value = _mm512_mask_blend_ps(_mm512_cmp_epi32_mask(_s_total, _mm512_set1_epi32(95), _MM_CMPINT_LE), _qres_dres_cos_value, _tmp6);
                //_qres_dres_cos_value = _mm512_mask_blend_ps(_mm512_cmp_epi32_mask(_s_total, _mm512_set1_epi32(79), _MM_CMPINT_LE), _qres_dres_cos_value, _tmp5);
                _qres_dres_cos_value = _mm512_mask_blend_ps(_mm512_cmp_epi32_mask(_s_total, _mm512_set1_epi32(63), _MM_CMPINT_LE), _qres_dres_cos_value, _tmp4);
                _qres_dres_cos_value = _mm512_mask_blend_ps(_mm512_cmp_epi32_mask(_s_total, _mm512_set1_epi32(47), _MM_CMPINT_LE), _qres_dres_cos_value, _tmp3);
                //_qres_dres_cos_value = _mm512_mask_blend_ps(_mm512_cmp_epi32_mask(_s_total, _mm512_set1_epi32(31), _MM_CMPINT_LE), _qres_dres_cos_value, _tmp2);
                //_qres_dres_cos_value = _mm512_mask_blend_ps(_mm512_cmp_epi32_mask(_s_total, _mm512_set1_epi32(15), _MM_CMPINT_LE), _qres_dres_cos_value, _tmp1);

                //_mm512_storeu_ps(ff.data(), _qres_dres_cos_value);
                //for(int j = 0; j < 16; j++) { std::cout<<ff[j]<<",";} std::cout<<std::endl; 
/*
                _mm512_storeu_si512(&zz[0], _high);

                for(int i = 0 ; i < 16;i++) {
                    uint32_t tmp = zz[i];
                    for (int j = 0; j < 32; j++) { 
                        std::cout<<(tmp & 1)<<",";
                        tmp >>= 1;                  
                    } std::cout<<std::endl;
                }
                std::vector<uint8_t> h(64);
                _mm512_storeu_si512(&h[0], _popcnt_lookup_table);
                for(int i = 0 ;i < 64;i++) {
                    uint8_t tmp = h[i];
                    std::cout<<unsigned(tmp)<<",";
                } std::cout<<std::endl;
                 

                std::cout<<"PUPUPUPUPUPU"<<std::endl;
                _mm512_storeu_si512(&h[0], _pl);

                for(int i = 0 ;i < 64;i++) {
                    uint8_t tmp = h[i];
                    std::cout<<unsigned(tmp)<<",";
                } std::cout<<std::endl;

                std::cout<<"PUPUPUPUPUPU"<<std::endl;
                _mm512_storeu_si512(&h[0], _ph);

                for(int i = 0 ;i < 64;i++) {
                    uint8_t tmp = h[i];
                    std::cout<<unsigned(tmp)<<",";
                } std::cout<<std::endl;
*/

                //__m512i _sum_result = _mm512_setzero_si512();
                //_sum_result = _mm512_adds_epu16(_sum_result, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(_s, 0)));
                //_sum_result = _mm512_adds_epu16(_sum_result, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(_s, 1)));
                //_s = _mm512_add_epi16(_mm512_and_si512(_s, _mask), _mm512_and_si512(_mm512_srli_epi16(_s, 4), _mask));

/*                std::vector<uint32_t> h(16);
                std::vector<uint8_t> h2(32);
                _mm512_storeu_si512(&h2[0], _s);
                for(int i = 0 ;i < 64;i++) {
                    uint8_t tmp = h2[i];
                    std::cout<<unsigned(tmp)<<",";
                } std::cout<<std::endl;
*/
/*
                std::vector<uint32_t> h(16);
                _mm512_storeu_si512(&h[0], _s);
                for (int i = 0 ; i < 16; i++) {
                    uint32_t tmp = h[i];
                    std::cout<<unsigned(tmp)<<",";
                } std::cout<<std::endl;
*/
                //std::vector<uint32_t> h(16);
                //_mm512_storeu_si512(&h[0], _s);
                //for (int i = 0 ; i < 16; i++) {
                 //   hamming[i] = std::cos( h[i] * ANGLE );
                //}
/*
                _mm512_storeu_si512(&h[0], _s);
                for (int i = 0 ; i < 16; i++) {
                    uint32_t tmp = h[i];
                    std::cout<<unsigned(tmp)<<",";
                } std::cout<<std::endl;
                for (int i = 0 ; i < 16; i++) {
                    uint16_t kk = tmp[i];//std::cout<<tmp[i]<<",";                  
                    for(int j = 0; j < 16; j++) { 
                        std::cout<<(kk & 1)<<",";
                        kk >>= 1;
                    } std::cout<<std::endl;
                    //hamming[i] = _popcnt32(tmp[i]);
                    //std::cout<<hamming[i]<<",";
                    hamming[i] = std::cos( hamming[i] * ANGLE );
                } 
                std::cout<<std::endl;
*/
                // compute Projected Qres Dres IP, save the exact_distance - query_projection_coefficient * neighbor_center_dot_product into results first
                //__m512 _neighbor_residual_center_negative_dot_product = _mm512_loadu_ps(stored_info);
                //stored_info += (16 * sizeof(float));
                //_mm512_storeu_ps(&appx_result_ptr[0], _mm512_add_ps(_exact_l2_distance, _mm512_mul_ps(_query_center_projection_coefficient, _neighbor_residual_center_negative_dot_product)));
                //for (int r = 0; r < 16; r++) { 
                //    appx_result_ptr[r] = (do_dot_product_simd(query_residual_vector.data(), &codebook[codebook_index_ptr[r] * low_rank], low_rank)); 
                //}

                //__m512 _qres_dres_cos_value = (__m512) _mm512_loadu_si512(hamming.data());
                //_qres_dres_cos_value = _mm512_mul_ps(_qres_dres_cos_value, _angle);

                //_mm512_storeu_ps(&appx_result_ptr[0], _qres_dres_cos_value);
                //for (int r = 0; r <=  15; r++) { std::cout<<appx_result_ptr[r]<<",";  } std::cout<<std::endl;
                //_qres_dres_cos_value = _mm512_div_ps(_qres_dres_cos_value, _projected_qres_norm);
                //_qres_dres_cos_value = _mm512_add_ps(_mm512_mul_ps(_correct_scale, _qres_dres_cos_value), _correct_bias);
                //_qres_dres_cos_value = _mm512_fmadd_ps(_correct_scale, _qres_dres_cos_value, _correct_bias);
                __m512 _qres_dres_ip = _mm512_mul_ps(_qres_dres_cos_value, _mm512_mul_ps(_qres_norm, _neighbor_res_norm));
                __m512 _appx_l2 = _mm512_fmadd_ps(_minus2, _qres_dres_ip, _exact_l2_distance);
                // std::cout<<_appx_l2<<" ";

                // ====attention=====
                // _mm512_storeu_ps(&appx_result_ptr[0], _qres_dres_cos_value);
                // for (int r = 0; r <=  15; r++) { std::cout<<appx_result_ptr[r]<<",";  }

                //_mm512_storeu_ps(&appx_result_ptr[0], _appx_l2);
                _mm512_storeu_ps(&appx_result_ptr[0], _mm512_mask_blend_ps(_mm512_cmp_ps_mask(_appx_l2, _topk_ub_dist, _CMP_LT_OQ), _falseValue, _trueValue));
                
                float* tmp_output_result=(float*)appx_result_ptr;
                appx_result_ptr += 16;

            }
/*
            if (num_left_index > 0) {
                // compute |dres|^2 
                __m512 _neighbor_res_norm = _mm512_loadu_ps(stored_info);
                stored_info += neighboring_float_size;
                //__m512 _neighbor_res_squared_norm = _mm512_loadu_ps(stored_info);
                //__m512 _neighbor_res_norm = _mm512_sqrt_ps(_neighbor_res_squared_norm);
                //__m512 _neighbor_res_squared_norm = _mm512_mul_ps(_neighbor_res_norm, _neighbor_res_norm);
    //           for (int r = 0; r < num_left_index; r++) { std::cout<<reinterpret_cast<const float*>(stored_info)[r]<<" ";} std::cout<<std::endl;
                // compute |dres|^2 
                __m512 _neighbor_center_projection_coefficient = _mm512_loadu_ps(stored_info);
    //            for (int r = 0; r < num_left_index; r++) { std::cout<<reinterpret_cast<const float*>(stored_info)[r]<<" ";} std::cout<<std::endl;
                stored_info += neighboring_float_size;
                //const auto codebook_index_ptr = reinterpret_cast<const uint32_t*>(stored_info);     
                //for (int r = 0; r < num_left_index; r++) { 
                //    _mm_prefetch(&codebook[codebook_index_ptr[r] * low_rank], _MM_HINT_T0);
               // }
                __m512 _neighbor_res_squared_norm = _mm512_mul_ps(_neighbor_res_norm, _neighbor_res_norm);
                // compute |qproj - dproj|^2
                __m512 _qproj_dproj_diff = _mm512_sub_ps(_query_center_projection_coefficient, _neighbor_center_projection_coefficient);
                __m512 _qproj_dproj_l2 = _mm512_mul_ps(_mm512_mul_ps(_qproj_dproj_diff, _qproj_dproj_diff), _center_node_squared_norm); 
                // combine exact distances
                __m512 _exact_l2_distance = _qproj_dproj_l2 + _qres_squared_norm + _neighbor_res_squared_norm;
                //_mm512_storeu_ps(&appx_result_ptr[0], _mm512_mask_blend_ps(_mm512_cmp_ps_mask(_exact_l2_distance, _topk_ub_dist, _CMP_LT_OQ), _falseValue, _trueValue));
                //_mm512_storeu_ps(&appx_result_ptr[0], _exact_l2_distance);
                //std::cout<<num_left_index<<"   "<<neighbor_size<<std::endl;
                //for (int r = 0; r < num_left_index; r++) { std::cout<<r<<" : "<<appx_result_ptr[r]<<" ";} std::cout<<std::endl;
                // compute Projected Qres Dres IP, save the exact_distance - query_projection_coefficient * neighbor_center_dot_product into results first
                //__m512 _neighbor_residual_center_negative_dot_product = _mm512_loadu_ps(stored_info);
                //for (int r = 0; r < num_left_index; r++) { std::cout<<reinterpret_cast<const float*>(stored_info)[r]<<" ";} std::cout<<std::endl;
                //stored_info += (16 * sizeof(float));
                //_mm512_storeu_ps(&appx_result_ptr[0], _mm512_add_ps(_exact_l2_distance, _mm512_mul_ps(_query_center_projection_coefficient, _neighbor_residual_center_negative_dot_product)));
                //std::cout<<num_left_index<<"  "<<rounds<<" "<<low_rank<<std::endl; 
                //_mm_prefetch(&codebook[codebook_index_ptr[0] * low_rank], _MM_HINT_T0);

                //for (int r = 0; r < num_left_index; r++) { 
                    //_mm_prefetch(&codebook[codebook_index_ptr[std::min(r + 1, num_left_index - 1)] * low_rank], _MM_HINT_T0);
                //    appx_result_ptr[r] = (do_dot_product_simd(query_residual_vector.data(), &codebook[codebook_index_ptr[r] * low_rank], low_rank));
    //                std::cout<<r<<" : "<<appx_result_ptr[r]<<" : "<<codebook_index_ptr[r]<<"  "<<do_dot_product_simd(query_residual_vector.data(), &codebook[codebook_index_ptr[r] * low_rank], low_rank)<<std::endl;
//                    std::cout<<appx_result_ptr[r]<<",";
                //}
                //std::cout<<std::endl;
                __m512 _qres_dres_cos_value = _mm512_loadu_ps(appx_result_ptr);
                //_qres_dres_cos_value = _mm512_add_ps(_mm512_mul_ps(_correct_scale, _qres_dres_cos_value), _correct_bias);
                __m512 _qres_dres_ip = _mm512_mul_ps(_qres_dres_cos_value, _mm512_mul_ps(_qres_norm, _neighbor_res_norm));
                __m512 _appx_l2 = _mm512_add_ps(_exact_l2_distance, _mm512_mul_ps(_minus2, _qres_dres_ip));
                
    
    
                _mm512_storeu_ps(&appx_result_ptr[0], _mm512_mask_blend_ps(_mm512_cmp_ps_mask(_appx_l2, _topk_ub_dist, _CMP_LT_OQ), _falseValue, _trueValue));
                //_mm512_storeu_ps(&appx_result_ptr[0], _appx_l2);
//                for (int r = 0; r <=  num_left_index - 1; r++) { std::cout<<appx_result_ptr[r]<<",";  } std::cout<<std::endl;
                
                // if the appx distance is larger than current bound
      //          for (int r = 0; r <=  num_left_index - 1; r++) { 
       //             std::cout<<r<<" : "<<appx_result_ptr[r]<<" : "<<codebook_index_ptr[r]<<"  "<<do_dot_product_simd(query_residual_vector.data(), &codebook[codebook_index_ptr[r] * low_rank], low_rank)<<std::endl;
        //        }
      
            }            
           */ 
        }

        // inline void compute_centroids(pecos::drm_t& X, int dsub, size_t ksub, uint32_t *assign, float *centroids, int threads=1) {
        //     // zero initialization for later do_axpy
        //     memset(centroids, 0, ksub * dsub * sizeof(*centroids));
        //     std::vector<float> centroids_size(ksub);
        //     #pragma omp parallel num_threads(threads)
        //     {
        //         // each thread takes care of [c_l, c_r)
        //         int rank = omp_get_thread_num();
        //         size_t c_l = (ksub * rank) / threads;
        //         size_t c_r = (ksub * (rank + 1)) / threads;
        //         for (size_t i = 0; i < X.rows; i++) {
        //             auto ci = assign[i];
        //             if (ci >= c_l && ci < c_r) {
        //                 float* y = centroids + ci * dsub;
        //                 const auto& xi = X.get_row(i);
        //                 pecos::do_axpy(1.0, xi.val, y, dsub);
        //                 centroids_size[ci] += 1;
        //             }
        //         }
        //         // normalize center vector
        //         for (size_t ci = c_l; ci < c_r; ci++) {
        //             float* y = centroids + ci * dsub;
        //             pecos::do_scale(1.0 / centroids_size[ci], y, dsub);
        //         }
        //     }
        // }

//         inline void train(std::vector<dist_t>& X_trn, std::vector<uint32_t>& encoded_result, std::vector<size_t> idx, uint32_t n_data, size_t dimension, uint32_t num_cluster_centroids, size_t sub_sample_points=0, int seed=0, size_t max_iter=10, int threads=32) {
           

//             std::srand(seed);
//             num_codebooks = num_cluster_centroids; 
//             codebook.resize(num_cluster_centroids * dimension);
//             encoded_result.resize(n_data, 0);
//             for (int i = 0; i < num_codebooks; i++) {
//                 size_t index = idx[i];
//                 memcpy(&codebook[i * low_rank], &X_trn[index * low_rank], sizeof(float) * low_rank);
//             }    
//             int repeat_times = num_codebooks / threads; 
//             for (int r = 0; r < repeat_times; r++) {
//                 #pragma omp parallel num_threads(threads)
//                 {
//                     int rank = omp_get_thread_num();
//                     int id = r * threads + rank;
//                     size_t index = idx[id];
//                     memcpy(&codebook[id * low_rank], &X_trn[index * low_rank], sizeof(float) * low_rank);
//                 }
//             }
//             for (int i = repeat_times * threads; i < num_codebooks; i++) {
//                 size_t index = idx[i];
//                 memcpy(&codebook[i * low_rank], &X_trn[index * low_rank], sizeof(float) * low_rank);
//             }

// /*
//             std::vector<size_t> indices(n_data, 0);
//             std::iota(indices.data(), indices.data() + n_data, 0);
//             std::random_shuffle(indices.data(), indices.data() + n_data);
//             for (size_t i = 0; i < num_cluster_centroids; i++) {
//                 size_t index = indices[i];
//                 memcpy(&codebook[i * low_rank], &X_trn[index * low_rank], sizeof(float) * low_rank);
//             }
//             // normalize codebook
//             for (int i = 0; i < num_codebooks; i++) {
//                 float norm = do_dot_product_simd(&codebook[i * low_rank], &codebook[i * low_rank], low_rank);
//                 norm = std::sqrt(norm);
//                 for (int j = 0; j < low_rank; j++) {
//                     codebook[i * low_rank + j] /= norm;
//                 }
//             }

//             std::vector<float> centroids;
//             num_codebooks = num_cluster_centroids; 
//             codebook.resize(num_cluster_centroids * dimension);
//             encoded_result.resize(n_data, 0);
//             std::generate(codebook.begin(), codebook.end(), std::rand);
//             // normalize codebook
//             for (int i = 0; i < num_codebooks; i++) {
//                 float norm = do_dot_product_simd(&codebook[i * low_rank], &codebook[i * low_rank], low_rank);
//                 norm = std::sqrt(norm);
//                 for (int j = 0; j < low_rank; j++) {
//                     codebook[i * low_rank + j] /= norm;
//                 }
//             }
//             int repeat_times = n_data / threads; 
//             for (int r = 0; r < repeat_times; r++) {
//                 #pragma omp parallel num_threads(threads)
//                 {
//                     int rank = omp_get_thread_num();
//                     encoded_result[r * threads + rank] = encode(&X_trn[(r * threads + rank) * low_rank]); //do_l2_distance_simd(vector, &codebook[(r * threads + rank) * low_rank], low_rank);
//                 }
//             }
//             for (int i = repeat_times * threads; i < n_data; i++) {
//                 encoded_result[i] = encode(&X_trn[i * low_rank]);
//             }
// */

// /*
//             if (sub_sample_points == 0) {
//                 sub_sample_points = n_data;
//             }



//             std::vector<float> xslice(sub_sample_points * dimension);
//             std::vector<size_t> indices(n_data, 0);
//             std::iota(indices.data(), indices.data() + n_data, 0);
//             std::random_shuffle(indices.data(), indices.data() + n_data);
//             for (size_t i = 0; i < sub_sample_points; i++) {
//                 size_t index = indices[i];
//                 std::memcpy(xslice.data() + i * dimension, &X_trn[index * dimension], dimension * sizeof(float));
//             }

//             pecos::drm_t Xsub;
//             Xsub.rows = sub_sample_points;
//             Xsub.cols = dimension;
//             Xsub.val = xslice.data();

//             // fit HLT or flat-Kmeans for each sub-space
//             std::vector<uint32_t> assignments(sub_sample_points);
//             pecos::clustering::Tree hlt(std::log2(num_cluster_centroids));
//             hlt.run_clustering<pecos::drm_t, uint32_t>(
//                 Xsub,
//                 0,
//                 seed,
//                 assignments.data(),
//                 max_iter,
//                 threads);
//             std::copy(assignments.begin(), assignments.end(), encoded_result.begin()); 
//             compute_centroids(Xsub, dimension, num_cluster_centroids, assignments.data(),
//               &codebook[0], threads);
//             // normalize codebook
//             int repeat_times = num_codebooks / threads; 
//             for (int r = 0; r < repeat_times; r++) {
//                 #pragma omp parallel num_threads(threads)
//                 {
//                     int rank = omp_get_thread_num();
//                     int codebook_num = r * threads + rank;
//                     float norm = do_dot_product_simd(&codebook[codebook_num * low_rank], &codebook[codebook_num * low_rank], low_rank);
//                     norm = std::sqrt(norm);
//                     for (int j = 0; j < low_rank; j++) {
//                         codebook[codebook_num * low_rank + j] /= norm;
//                     }
//                 }
//             }
//             for (int i = repeat_times * threads; i < num_codebooks; i++) {
//                 int codebook_num = i;
//                 float norm = do_dot_product_simd(&codebook[codebook_num * low_rank], &codebook[codebook_num * low_rank], low_rank);
//                 norm = std::sqrt(norm);
//                 for (int j = 0; j < low_rank; j++) {
//                     codebook[i * low_rank + j] /= norm;
//                 }
//             }
// */
//             // learn HNSW encoder
//             //pecos::drm_t X;
//             //X.rows = num_codebooks;
//             //X.cols = low_rank;
//             //X.val = codebook.data();
//             //pecos::ann::HNSW<float, feat_vec_t> encoder;
//             //encoder.train(X, 64, 500, threads, 8);

//     // inference
//     //     uint32_t num_data = X_tst.rows;
//     //         auto searcher = indexer.create_searcher(); 

// /*
//             for (int i = 0; i < num_codebooks; i++) {
//                 float norm = do_dot_product_simd(&codebook[i * low_rank], &codebook[i * low_rank], low_rank);
//                 norm = std::sqrt(norm);
//                 for (int j = 0; j < low_rank; j++) {
//                     codebook[i * low_rank + j] /= norm;
//                 }
//             }
// */

//             repeat_times = n_data / threads; 
//             for (int r = 0; r < repeat_times; r++) {
//                 #pragma omp parallel num_threads(threads)
//                 {
//                     int rank = omp_get_thread_num();
//                     encoded_result[r * threads + rank] = encode(&X_trn[(r * threads + rank) * low_rank]); //do_l2_distance_simd(vector, &codebook[(r * threads + rank) * low_rank], low_rank);
//                 }
//             }
//             for (int i = repeat_times * threads; i < n_data; i++) {
//                 encoded_result[i] = encode(&X_trn[i * low_rank]);
//             }

//         }

        // inline uint32_t encode(float* vector) {
        //     pecos::drm_t X;
        //     X.rows = 1;
        //     X.cols = low_rank;
        //     X.val = vector;
            //auto searcher = encoder.create_searcher();
            //auto ret_pairs = encoder.predict_single(X.get_row(0), 50, 1,searcher);
             //max_heap_t& predict_single(const feat_vec_t& query, uint32_t efS, uint32_t topk, Searcher& searcher) const {
            //for (auto dist_idx_pair : ret_pairs) {
            //    return dist_idx_pair.node_id;
            //}
//            std::vector<float> dist(num_codebooks, 0);
//            for (int i = 0; i < num_codebooks; i++) {
//                dist[i] = do_l2_distance_simd(vector, &codebook[i * low_rank], low_rank);
//            }
//            for( int i = 0; i < low_rank; i++)
 //                { std::cout<<vector[i]<<","; } std::cout<<std::endl;
  //          for( int i = 0; i < low_rank; i++)
   //              { std::cout<<codebook[0 + i]<<","; } std::cout<<std::endl;

//                std::cout<<dist[0]<<" "<<std::endl;;
            //    dist[i] = do_l2_distance_simd(vector, &codebook[i * low_rank], low_rank);
              
/*
            int repeat_times = num_codebooks / threads; 
            
            for(int r = 0; r < repeat_times; r++) {
        
                #pragma omp parallel num_threads(threads)
                {
                    int rank = omp_get_thread_num();
                    dist[r * threads + rank] = do_l2_distance_simd(vector, &codebook[(r * threads + rank) * low_rank], low_rank);
                }
            }
            for (int i = repeat_times * threads; i < num_codebooks; i++) {
                dist[i] = do_l2_distance_simd(vector, &codebook[i * low_rank], low_rank);
            }
*/ 
//            std::vector<float>::iterator argmin_result = std::min_element(dist.begin(), dist.end());
//            return std::distance(dist.begin(), argmin_result);
        // }

    };


}  // end of namespace hnswlib

