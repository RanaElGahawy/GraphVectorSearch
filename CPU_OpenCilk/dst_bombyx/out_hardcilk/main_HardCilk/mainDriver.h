#pragma once

#include <hardCilkDriver.h>
#include "main_defs.h"

class dst_searchDriver : public hardCilkDriver {
public:
    dst_searchDriver(Memory *memory, int argc, char *argv[]) : hardCilkDriver(memory) {}

    int run_test_bench() override {
        dst_search_task root_task_0 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        int counter = 2;

        uint64_t addr = allocateMemFPGA(sizeof(root_task_0), sizeof(root_task_0));
        memory_->copyToDevice(addr, reinterpret_cast<const uint8_t *>(&root_task_0), sizeof(root_task_0));
        memory_->copyToDevice(addr, reinterpret_cast<const uint8_t *>(&counter), sizeof(counter));

        int slot_size = mg * mc * maxDegree;

        uint64_t vecs_addr = allocateMemFPGA(sizeof(const float) * , 512);
        memory_->copyToDevice(vecs_addr, reinterpret_cast<const uint8_t *>(vecs), sizeof(const float) * );
        uint64_t degree_addr = allocateMemFPGA(sizeof(const int) * , 512);
        memory_->copyToDevice(degree_addr, reinterpret_cast<const uint8_t *>(degree), sizeof(const int) * );
        uint64_t links_addr = allocateMemFPGA(sizeof(const int) * , 512);
        memory_->copyToDevice(links_addr, reinterpret_cast<const uint8_t *>(links), sizeof(const int) * );

        root_task_0._cont = addr;
        root_task_0.vecs = vecs_addr;
        root_task_0.degree = degree_addr;
        root_task_0.links = links_addr;
        root_task_0.d = d;
        root_task_0.maxDegree = maxDegree;
        root_task_0.entry = entry;
        root_task_0.ef = ef;
        root_task_0.cand_cap = cand_cap;
        root_task_0.mc = mc;
        root_task_0.mg = mg;
        root_task_0.k0 = k;
        root_task_0._padding = 0;

        startSystem();

        auto start_management = std::chrono::high_resolution_clock::now();
        managementLoop();
        auto end_management = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> management_duration = end_management - start_management;
        std::cout << "Time taken by management_loop: " << management_duration.count() << " seconds" << std::endl;


        cilk_sync;

        return 0;
    }
};
