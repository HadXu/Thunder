#ifndef DLSYS_RUNTIME_DEVICE_API_H_
#define DLSYS_RUNTIME_DEVICE_API_H_

#include "c_runtime_api.h"
#include <assert.h>
#include <string>


namespace dlsys {
    namespace runtime {
        class DeviceAPI{
            public:
                virtual ~DeviceAPI(){}
                virtual void *AllocDataSpace(DLContext ctx, size_t size,
                                         size_t alignment) = 0;
                virtual void FreeDataSpace(DLContext ctx, void *ptr) = 0;
                virtual void CopyDataFromTo(const void *from, void *to, size_t size,
                                        DLContext ctx_from, DLContext ctx_to,
                                        DLStreamHandle stream) = 0;
                 virtual void StreamSync(DLContext ctx, DLStreamHandle stream) = 0;
        };
    }
}

#endif