/*
 * Copyright 2018-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef NV_INFER_SERIALIZE_H
#define NV_INFER_SERIALIZE_H

#include "NvInfer.h"

namespace nvinfer1
{
namespace serialize
{

//!
//! \brief Serialize TensorRT INetworkDefinition.
//! \param network Network to serialize.
//!
//! \return A nvinfer1::IHostMemory object that contains serialized network.
//!
//! The network might be deserialized with nvserialize::deserializeINetwork.
//!
//! \see nvinfer1::serialize::deserializeNetwork
//!
TENSORRTAPI nvinfer1::IHostMemory* serializeNetwork(const nvinfer1::INetworkDefinition* network);

class IDeserializer
{
public:
    //!
    //! \brief Deserialize TensorRT INetworkDefinition from stream.
    //! \param blob The memory that holds the serialized network.
    //! \param size The size of the memory.
    //! \return True iff deserialization was successful.
    //!
    virtual bool deserialize(const void* blob, std::size_t size) TRTNOEXCEPT = 0;

    //!
    //! \brief Destroy the IDeserializer object.
    //!
    virtual void destroy() TRTNOEXCEPT = 0;

    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. Setting 
    //! recorder to nullptr unregisters the recorder with the interface, resulting in a call to decRefCount if
    //! a recorder has been registered.
    //! 
    //! \param recorder The error recorder to register with this interface.
    //
    //! \see getErrorRecorder
    //! 
    virtual void setErrorRecorder(IErrorRecorder* recorder) TRTNOEXCEPT = 0;

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A default error recorder does not exist,
    //! so a nullptr will be returned if setErrorRecorder has not been called.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder
    //!
    virtual IErrorRecorder* getErrorRecorder() const TRTNOEXCEPT = 0;
};

//!
//! \brief Creates IDeserializer object.
//! \param network A nvinfer1::INetworkDefinition object where deserialized network will be placed.
//!
TENSORRTAPI IDeserializer* createDeserializer(nvinfer1::INetworkDefinition* network, nvinfer1::ILogger* logger);

} // namespace serialize
} // namespace nvinfer1

#endif // NV_INFER_SERIALIZE_H