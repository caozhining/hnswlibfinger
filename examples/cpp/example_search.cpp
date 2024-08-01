#include "../../hnswlib/hnswlib.h"
#include <fcntl.h>
#include <unistd.h>

int groundtruth_num = 100;

void load_data(const char *filename, float *&data, int &num,
               int &dim)
{ // load data with sift10K/sift1M/gist1M pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "Error: open file! " << filename << std::endl;
        exit(-1);
    }
    else
    {
        std::cout << "Data loading from " << filename << std::endl;
    }
    in.read((char *)&dim, 4);
    std::cout << "Data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    std::cout << "Data quantity: " << num << std::endl;
    data = new float[num * dim];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++)
    {
        in.seekg(4, std::ios::cur);
        in.read((char *)(data + i * dim), dim * 4);
    }
    in.close();
    std::cout << "Data loading completed!" << std::endl;
}

void load_data_groundtruth(const char *filename, unsigned int *&data, int num)
{ // load data with sift10K/sift1M/gist1M pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cout << "Error: open file! " << filename << std::endl;
        exit(-1);
    }
    else
    {
        std::cout << "Data loading from " << filename << std::endl;
    }
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;

    if ((unsigned)(fsize / (groundtruth_num + 1) / 4) != num)
    {
        std::cout << "Error: file size!" << std::endl;
        exit(-1);
    }
    else
    {
        std::cout << "Data quantity: " << num << std::endl;
    };

    data = (unsigned int *)new char[num * groundtruth_num * sizeof(unsigned int)];

    in.seekg(0, std::ios::beg);
    unsigned int temp;
    for (size_t i = 0; i < num; i++)
    {
        in.read((char *)&temp, 4);
        // std::cout<<temp<<" ";
        if (temp != groundtruth_num)
        {
            std::cout << "Error: temp value!" << std::endl;
            exit(-1);
        }
        in.read((char *)(data + i * groundtruth_num), temp * 4);
    }
    in.close();
    std::cout << "Data loading completed!" << std::endl;
}

int main(int argc, char **argv) {
    int dim = 16;               // Dimension of the elements
    int num_base_elements;      // Maximum number of elements, should be known beforehand
    int max_elements = 10000;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 128;  // Controls index search speed/build speed tradeoff

    int candidate_array_size = atof(argv[1]);

    float *base_data = NULL;    // Pointer to loading base data
    std::string base_data_path;
    std::string query_data_path;
    std::string groundtruth_data_path;
    std::string hnsw_save_path;

    base_data_path = ("/home/zhining/zjuczn/hnswlib-master/sift/sift_base.fvecs");
    query_data_path = ("/home/zhining/zjuczn/hnswlib-master/sift/sift_query.fvecs");
    groundtruth_data_path = ("/home/zhining/zjuczn/hnswlib-master/sift/sift_groundtruth.ivecs");
    hnsw_save_path = ("/home/zhining/zjuczn/hnswlib-master/sift/SIFT1M_FULL_Base_M_" + std::to_string(M) + "_efConstruction_" + std::to_string(ef_construction) + ".bin"); // Path to the backup of index


    load_data(base_data_path.c_str(), base_data, num_base_elements, dim);


    std::cout<<candidate_array_size<<"\n";

#ifdef __AVX__
    std::cout<<"Open AVX\n";
#ifdef __AVX512F__
    std::cout<<"Open AVX512\n";
#endif
#endif

    // Initing index
    hnswlib::L2Space space(dim);


    
    // hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // // int flag = true;
    // std::cout << "Read index from " << hnsw_save_path << std::endl;
    // hnswlib::HierarchicalNSWFinger<float>* alg_hnsw = new hnswlib::HierarchicalNSWFinger<float>(&space, hnsw_save_path);
    
    // hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_save_path);
    // hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, num_base_elements, M, ef_construction);
 
    hnswlib::HierarchicalNSWFinger<float>* alg_hnsw = new hnswlib::HierarchicalNSWFinger<float>(&space, num_base_elements, M, ef_construction);
    for (int i = 0; i < num_base_elements; i++)
    {
        if (i % 50000 == 0)
        {
            std::cout << i << std::endl;
        }
        alg_hnsw->addPoint(base_data + i * dim, i);
    }

    alg_hnsw->CreateFinger();

    std::cout<<"finish create finger\n";

    // std::cout<<"start save index\n";

    // alg_hnsw->saveIndex(hnsw_save_path);

    // std::cout<<"finish save index\n";

    int num_query_elements;
    float *query_data = NULL;
    load_data(query_data_path.c_str(), query_data, num_query_elements, dim);


    // float *dddd=base_data;
    // for(int i=0;i<dim;i++)
    //     std::cout<<dddd[i]<<" ";std::cout<<"\n";

    // // load groundtruth data from storage
    unsigned int *groundtruth_data = NULL;
    load_data_groundtruth(groundtruth_data_path.c_str(), groundtruth_data, num_query_elements);

    alg_hnsw->setup_appx_results_containers();
    int candidate_size_array[30]={10,11,12,13,15,20,25,30,40,50,65,80,100,200,300,500,750,1000,1500,2000,3000,4000,5000,6000};
    std::cout<<"test begin:\n";
    for(int x = 0; x< 23; x++)
    {
        int now_candidate_size=candidate_size_array[x];
        hnswlib::totcalctime=0;
        std::chrono::high_resolution_clock::time_point KNNStart = std::chrono::high_resolution_clock::now();
        // Query the elements for themselves and measure recall
        float recall = 0;
        uint64_t correct;
        float average_query_latency = 0;
        int topK = 10;
        std::cout << "==================\n";
        std::cout << "topK = " << topK << std::endl;
        std::cout << "candidate size = "<<now_candidate_size<<"\n";
        assert(topK <= now_candidate_size);

        correct = 0; // the number of correct nodes in the result queue.

        // alg_hnsw->setup_appx_results_containers();

        for (int i = 0; i < num_query_elements; i++)
        {
            // std::cout<<"assignment "<<i<<"\n";
            // NOTE !!! result中的距离是负数，便于排序
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(query_data + i * dim, now_candidate_size);

            // while (result.size())

            int previous_id = -1;
            for (int kk = 0; kk < topK; kk++) // 判定result的前topK个在不在groundtruth_data的前topK个里
            {
                // 判断result中是否有重复的点
                if (previous_id == result.top().second)
                {
                    std::cout << "Iteration: " << kk << " Repeat Node ID: " << previous_id << " " << result.top().second << std::endl;
                    assert(previous_id != result.top().second);
                }
                previous_id = result.top().second;

                int* groundtruth_float= (int*)(groundtruth_data + i * groundtruth_num);

                // for(int j = 0; j <topK ;j++)
                //     std::cout<<groundtruth_float[j]<<" ";
                // std::cout<<"\n";
                // std::cout<<"our "<<result.top().second<<" : "<<result.top().first<<" \n";

                // std::cout << result.top().first << " " << result.top().second << std::endl;
                if (std::find(groundtruth_data + i * groundtruth_num, groundtruth_data + i * groundtruth_num + topK, result.top().second) != groundtruth_data + i * groundtruth_num + topK)
                {
                    correct++;
                    // std::cout<<"correct"<<"\n";
                }
                result.pop();
            }
        }
        recall = correct * 1.0 / topK / num_query_elements;
        std::cout << "Avarage recall: " << recall*100 << std::endl;
        std::chrono::high_resolution_clock::time_point KNNEnd = std::chrono::high_resolution_clock::now();
        auto KNNTime = std::chrono::duration_cast<std::chrono::nanoseconds>(KNNEnd - KNNStart).count();
        float KNNfloattime=((float)KNNTime)/1000000000;
        std::cout << "Time: "<<KNNfloattime << " s\n";
        std::cout << "Dist calc time "<<hnswlib::totcalctime<<"\n";
        std::cout << "==================\n";
    }

    // // Serialize index
    // std::string hnsw_path = "hnsw.bin";
    // alg_hnsw->saveIndex(hnsw_path);
    // delete alg_hnsw;

    // // Deserialize index and check recall
    // alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
    // correct = 0;
    // for (int i = 0; i < max_elements; i++) {
    //     std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
    //     hnswlib::labeltype label = result.top().second;
    //     if (label == i) correct++;
    // }
    // recall = (float)correct / max_elements;
    // std::cout << "Recall of deserialized index: " << recall << "\n";

    // delete[] data;
    // delete alg_hnsw;
    return 0;
}
