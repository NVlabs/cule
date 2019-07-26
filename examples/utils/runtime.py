import ctypes
import enum
import os
import sys

class _Structure(ctypes.Structure):
  def cptr(self):
    return ctypes.pointer(self)

  def __str__(self):
    strs = []
    for k in [m for m in dir(self) if (m[0] != '_') and (m != 'cptr')]:
      s = getattr(self, k)
      if 'c_int_Array' in type(s).__name__:
        s = [s[i] for i in range(len(s))]
      elif isinstance(s, bytes):
        try:
          s = s.decode(sys.stdout.encoding)
        except:
          pass
      strs.append((k, s))
    return '\n'.join(['{}: {}'.format(i, j) for i, j in strs])

class cudaError_t(enum.IntEnum):
  cudaSuccess = 0
  cudaErrorMissingConfiguration = 1
  cudaErrorMemoryAllocation = 2
  cudaErrorInitializationError = 3
  cudaErrorLaunchFailure = 4
  cudaErrorPriorLaunchFailure = 5
  cudaErrorLaunchTimeout = 6
  cudaErrorLaunchOutOfResources = 7
  cudaErrorInvalidDeviceFunction = 8
  cudaErrorInvalidConfiguration = 9
  cudaErrorInvalidDevice = 10
  cudaErrorInvalidValue = 11
  cudaErrorInvalidPitchValue = 12
  cudaErrorInvalidSymbol = 13
  cudaErrorMapBufferObjectFailed = 14
  cudaErrorUnmapBufferObjectFailed = 15
  cudaErrorInvalidHostPointer = 16
  cudaErrorInvalidDevicePointer = 17
  cudaErrorInvalidTexture = 18
  cudaErrorInvalidTextureBinding = 19
  cudaErrorInvalidChannelDescriptor = 20
  cudaErrorInvalidMemcpyDirection = 21
  cudaErrorAddressOfConstant = 22
  cudaErrorTextureFetchFailed = 23
  cudaErrorTextureNotBound = 24
  cudaErrorSynchronizationError = 25
  cudaErrorInvalidFilterSetting = 26
  cudaErrorInvalidNormSetting = 27
  cudaErrorMixedDeviceExecution = 28
  cudaErrorCudartUnloading = 29
  cudaErrorUnknown = 30
  cudaErrorNotYetImplemented = 31
  cudaErrorMemoryValueTooLarge = 32
  cudaErrorInvalidResourceHandle = 33
  cudaErrorNotReady = 34
  cudaErrorInsufficientDriver = 35
  cudaErrorSetOnActiveProcess = 36
  cudaErrorInvalidSurface = 37
  cudaErrorNoDevice = 38
  cudaErrorECCUncorrectable = 39
  cudaErrorSharedObjectSymbolNotFound = 40
  cudaErrorSharedObjectInitFailed = 41
  cudaErrorUnsupportedLimit = 42
  cudaErrorDuplicateVariableName = 43
  cudaErrorDuplicateTextureName = 44
  cudaErrorDuplicateSurfaceName = 45
  cudaErrorDevicesUnavailable = 46
  cudaErrorInvalidKernelImage = 47
  cudaErrorNoKernelImageForDevice = 48
  cudaErrorIncompatibleDriverContext = 49
  cudaErrorPeerAccessAlreadyEnabled = 50
  cudaErrorPeerAccessNotEnabled = 51
  cudaErrorDeviceAlreadyInUse = 54
  cudaErrorProfilerDisabled = 55
  cudaErrorProfilerNotInitialized = 56
  cudaErrorProfilerAlreadyStarted = 57
  cudaErrorProfilerAlreadyStopped = 58
  cudaErrorAssert = 59
  cudaErrorTooManyPeers = 60
  cudaErrorHostMemoryAlreadyRegistered = 61
  cudaErrorHostMemoryNotRegistered = 62
  cudaErrorOperatingSystem = 63
  cudaErrorPeerAccessUnsupported = 64
  cudaErrorLaunchMaxDepthExceeded = 65
  cudaErrorLaunchFileScopedTex = 66
  cudaErrorLaunchFileScopedSurf = 67
  cudaErrorSyncDepthExceeded = 68
  cudaErrorLaunchPendingCountExceeded = 69
  cudaErrorNotPermitted = 70
  cudaErrorNotSupported = 71
  cudaErrorHardwareStackError = 72
  cudaErrorIllegalInstruction = 73
  cudaErrorMisalignedAddress = 74
  cudaErrorInvalidAddressSpace = 75
  cudaErrorInvalidPc = 76
  cudaErrorIllegalAddress = 77
  cudaErrorInvalidPtx = 78
  cudaErrorInvalidGraphicsContext = 79
  cudaErrorNvlinkUncorrectable = 80
  cudaErrorJitCompilerNotFound = 81
  cudaErrorCooperativeLaunchTooLarge = 82
  cudaErrorSystemNotReady = 83
  cudaErrorIllegalState = 84
  cudaErrorStartupFailure = 127
  cudaErrorStreamCaptureUnsupported = 900
  cudaErrorStreamCaptureInvalidated = 901
  cudaErrorStreamCaptureMerge = 902
  cudaErrorStreamCaptureUnmatched = 903
  cudaErrorStreamCaptureUnjoined = 904
  cudaErrorStreamCaptureIsolation = 905
  cudaErrorStreamCaptureImplicit = 906
  cudaErrorCapturedEvent = 907
  cudaErrorApiFailureBase = 10000

class CUuuid(_Structure):
  _fields_ = [('bytes', ctypes.c_char * 16)]

class cudaDeviceProp(_Structure):
  _fields_ = [('name', ctypes.c_char * 256), ('uuid', CUuuid), ('luid', ctypes.c_char * 8), ('luidDeviceNodeMask', ctypes.c_uint), ('totalGlobalMem', ctypes.c_ulong), ('sharedMemPerBlock', ctypes.c_ulong), ('regsPerBlock', ctypes.c_int), ('warpSize', ctypes.c_int), ('memPitch', ctypes.c_ulong), ('maxThreadsPerBlock', ctypes.c_int), ('maxThreadsDim', ctypes.c_int * 3), ('maxGridSize', ctypes.c_int * 3), ('clockRate', ctypes.c_int), ('totalConstMem', ctypes.c_ulong), ('major', ctypes.c_int), ('minor', ctypes.c_int), ('textureAlignment', ctypes.c_ulong), ('texturePitchAlignment', ctypes.c_ulong), ('deviceOverlap', ctypes.c_int), ('multiProcessorCount', ctypes.c_int), ('kernelExecTimeoutEnabled', ctypes.c_int), ('integrated', ctypes.c_int), ('canMapHostMemory', ctypes.c_int), ('computeMode', ctypes.c_int), ('maxTexture1D', ctypes.c_int), ('maxTexture1DMipmap', ctypes.c_int), ('maxTexture1DLinear', ctypes.c_int), ('maxTexture2D', ctypes.c_int * 2), ('maxTexture2DMipmap', ctypes.c_int * 2), ('maxTexture2DLinear', ctypes.c_int * 3), ('maxTexture2DGather', ctypes.c_int * 2), ('maxTexture3D', ctypes.c_int * 3), ('maxTexture3DAlt', ctypes.c_int * 3), ('maxTextureCubemap', ctypes.c_int), ('maxTexture1DLayered', ctypes.c_int * 2), ('maxTexture2DLayered', ctypes.c_int * 3), ('maxTextureCubemapLayered', ctypes.c_int * 2), ('maxSurface1D', ctypes.c_int), ('maxSurface2D', ctypes.c_int * 2), ('maxSurface3D', ctypes.c_int * 3), ('maxSurface1DLayered', ctypes.c_int * 2), ('maxSurface2DLayered', ctypes.c_int * 3), ('maxSurfaceCubemap', ctypes.c_int), ('maxSurfaceCubemapLayered', ctypes.c_int * 2), ('surfaceAlignment', ctypes.c_ulong), ('concurrentKernels', ctypes.c_int), ('ECCEnabled', ctypes.c_int), ('pciBusID', ctypes.c_int), ('pciDeviceID', ctypes.c_int), ('pciDomainID', ctypes.c_int), ('tccDriver', ctypes.c_int), ('asyncEngineCount', ctypes.c_int), ('unifiedAddressing', ctypes.c_int), ('memoryClockRate', ctypes.c_int), ('memoryBusWidth', ctypes.c_int), ('l2CacheSize', ctypes.c_int), ('maxThreadsPerMultiProcessor', ctypes.c_int), ('streamPrioritiesSupported', ctypes.c_int), ('globalL1CacheSupported', ctypes.c_int), ('localL1CacheSupported', ctypes.c_int), ('sharedMemPerMultiprocessor', ctypes.c_ulong), ('regsPerMultiprocessor', ctypes.c_int), ('managedMemory', ctypes.c_int), ('isMultiGpuBoard', ctypes.c_int), ('multiGpuBoardGroupID', ctypes.c_int), ('hostNativeAtomicSupported', ctypes.c_int), ('singleToDoublePrecisionPerfRatio', ctypes.c_int), ('pageableMemoryAccess', ctypes.c_int), ('concurrentManagedAccess', ctypes.c_int), ('computePreemptionSupported', ctypes.c_int), ('canUseHostPointerForRegisteredMem', ctypes.c_int), ('cooperativeLaunch', ctypes.c_int), ('cooperativeMultiDeviceLaunch', ctypes.c_int), ('sharedMemPerBlockOptin', ctypes.c_ulong), ('pageableMemoryAccessUsesHostPageTables', ctypes.c_int), ('directManagedMemAccessFromHost', ctypes.c_int)]

class Runtime(object):
  def __init__(self, path=None, lib='libcudart.so'):
    if not path:
      path = self._locate()['lib64']
    self._lib = ctypes.cdll.LoadLibrary(os.path.join(path, lib))
    self._populate()

  def _cuda_return_check(self, rc, function, args):
    if rc != 0:
      msg = self._lib.cudaGetErrorString(rc)
      raise RuntimeError('{}: {} [code {}]'.format(function.__name__, msg.decode(sys.stdout.encoding), cudaError_t(rc).name))
    return rc

  def _cimport(self, fn, argt, rest=cudaError_t, signature=None):
    fn.restype = rest
    fn.argtypes = argt
    fn.__signature__ = signature
    if rest == cudaError_t:
      fn.errcheck = self._cuda_return_check
    return fn

  def _find_in_path(self, name, path):
      for directory in path.split(os.pathsep):
          binpath = os.path.join(directory, name)
          if os.path.exists(binpath):
              return os.path.abspath(binpath)
      return None

  def _locate(self):
      if 'CUDAHOME' in os.environ:
          home = os.environ['CUDAHOME']
          nvcc = os.path.join(home, 'bin', 'nvcc')
      else:
          nvcc = self._find_in_path('nvcc', os.environ['PATH'])
          if nvcc is None:
              raise EnvironmentError('The nvcc binary could not be '
                                     'located in your $PATH. Either '
                                     'add it to your path, or set $CUDAHOME')
          home = os.path.dirname(os.path.dirname(nvcc))
      cudaconfig = {
              'home':home,
              'nvcc':nvcc,
              'include': os.path.join(home, 'include'),
              'lib64': os.path.join(home, 'lib64')
              }
      for k, v in cudaconfig.items():
          if not os.path.exists(v):
              raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))
      return cudaconfig

  def _populate(self):
    self.cudaGetDeviceCount = self._cimport(self._lib.cudaGetDeviceCount, [ctypes.POINTER(ctypes.c_int)], cudaError_t, '(count)')
    self.cudaGetDeviceProperties = self._cimport(self._lib.cudaGetDeviceProperties, [ctypes.POINTER(cudaDeviceProp), ctypes.c_int], cudaError_t, '(prop, device)')
    self.cudaMemGetInfo = self._cimport(self._lib.cudaMemGetInfo, [ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong)], cudaError_t, '(free, total)')

def get_device_props():
    num_devices = ctypes.c_int(0)

    rtlib = Runtime()
    rtlib.cudaGetDeviceCount(num_devices)
    props = [cudaDeviceProp()] * num_devices.value
    for i, p in enumerate(props):
        rtlib.cudaGetDeviceProperties(p, i)

    return props

def cuda_device_str(device=0):
    props = cudaDeviceProp()

    rtlib = Runtime()
    rtlib.cudaGetDeviceProperties(props, device)

    freeMem, totalMem = ctypes.c_ulong(), ctypes.c_ulong()
    rtlib.cudaMemGetInfo(freeMem, totalMem)
    memBandwidth = (props.memoryClockRate * 1000.0) * (props.memoryBusWidth / 8 * 2) / 1.0e9

    return  '{} : {:8.3f} Mhz   (Ordinal {})\n'\
            '{} SMs enabled. Compute Capability sm_{}{}\n'\
            'FreeMem: {:6,d}MB   TotalMem: {:6,d}MB   {:2d}-bit pointers.\n'\
            'Mem Clock: {:8.3f} Mhz x {} bits   ({:5.1f} GB/s)\n'\
            'ECC {}\n'.format(
                    props.name.decode(sys.stdout.encoding), props.clockRate / 1000.0, device,
                    props.multiProcessorCount, props.major, props.minor,
                    int(freeMem.value / (1 << 20)), int(totalMem.value / (1 << 20)), 8 * ctypes.sizeof(ctypes.POINTER(ctypes.c_int32)),
                    props.memoryClockRate / 1000.0, props.memoryBusWidth, memBandwidth,
                    'Enabled' if props.ECCEnabled else 'Disabled')
