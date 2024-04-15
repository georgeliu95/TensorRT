import argparse
import tempfile
import urllib.request
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--os", required=True, help="OS")
    parser.add_argument("--cuda", required=True, help="Cuda version")
    return parser.parse_args()

# This order needs to be preserved for dependency tracking.

DEB_PACKAGES_ROCKY=[
"libnvinfer10-10.0.1.5-1.cuda{ver}.x86_64.rpm",
"libnvonnxparsers10-10.0.1.5-1.cuda{ver}.x86_64.rpm",
"libnvinfer-plugin10-10.0.1.5-1.cuda{ver}.x86_64.rpm",
"libnvinfer-vc-plugin10-10.0.1.5-1.cuda{ver}.x86_64.rpm",
"libnvinfer-headers-devel-10.0.1.5-1.cuda{ver}.x86_64.rpm",
"libnvinfer-headers-plugin-devel-10.0.1.5-1.cuda{ver}.x86_64.rpm",
"libnvinfer-devel-10.0.1.5-1.cuda{ver}.x86_64.rpm",
"libnvonnxparsers-devel-10.0.1.5-1.cuda{ver}.x86_64.rpm",
"libnvinfer-plugin-devel-10.0.1.5-1.cuda{ver}.x86_64.rpm",
"python3-libnvinfer-10.0.1.5-1.cuda{ver}.x86_64.rpm",
]

DEB_PACKAGES_UBUNTU=[
    "libnvinfer10_10.0.1.5-1{ext}",
    "libnvonnxparsers10_10.0.1.5-1{ext}",
    "libnvinfer-plugin10_10.0.1.5-1{ext}",
    "libnvinfer-vc-plugin10_10.0.1.5-1{ext}",
    "libnvinfer-headers-dev_10.0.1.5-1{ext}",
    "libnvinfer-headers-plugin-dev_10.0.1.5-1{ext}",
    "libnvinfer-dev_10.0.1.5-1{ext}",
    "libnvonnxparsers-dev_10.0.1.5-1{ext}",
    "libnvinfer-plugin-dev_10.0.1.5-1{ext}",
    "python3-libnvinfer_10.0.1.5-1{ext}",
]

ROOT_URL = "http://cuda-repo/release-candidates/Libraries/TensorRT/v10.0/10.0.1.5-a01cd51e/"

def get_cuda_props(cuda_ver):
    assert len(cuda_ver) >= 4
    cuda = cuda_ver[:4]
    if (cuda == "12.4"):
        return cuda, cuda+"-r550"
    if (cuda == "12.2"):
        return cuda, cuda+"-r535"
    elif (cuda == "12.0"):
        return cuda, cuda+"-r525"
    elif (cuda == "11.8"):
        return cuda, cuda+"-r520"
    elif (cuda == "10.2"):
        return cuda, cuda+"-r440"
    else:
        print("Found unsupported CUDA version: " + cuda)
        exit(-1)

def get_arch_props(os):
    if os == "22.04" or os == "20.04":
        return "amd64", "Ubuntu{ver}-x64-manylinux_2_17".format(ver=os.replace(".", "_")), "deb", "+", "_", False
    elif os == "8":
        return "x86_64", "RHEL8_9-x64-manylinux_2_17", "rpm", ".", ".", True
    elif os == "9":
        return "x86_64", "RHEL9_3-x64-manylinux_2_17", "rpm", ".", ".", True
    elif os == "cross-sbsa":
        return "all", "Ubuntu20_04-aarch64-aarch64-manylinux_2_31", "deb", ".", ".", False
    else:
        print("Found unsupported OS: " + os)
        exit(-1)

if __name__ == "__main__":
    args = parse_arguments()
    cuda, cuda_url  = get_cuda_props(args.cuda)
    arch, url, ext, prefix, sep, is_rocky = get_arch_props(args.os)
    suffix = "{prefix}cuda{cuda}{sep}{arch}.{ext}".format(prefix=prefix, cuda=cuda, sep=sep, arch=arch, ext=ext)

    with tempfile.TemporaryDirectory() as tmp:
        URL = ROOT_URL + "{cuda}/{url}/{ext}/".format(cuda=cuda_url, url=url, ext=ext)

        if is_rocky:
            for package in DEB_PACKAGES_ROCKY:
                package = package.format(ver=cuda)
                full_url = URL+package
                print("Downloading from {URL}...".format(URL=full_url))
                try:
                    urllib.request.urlretrieve(full_url, package)
                except urllib.error.HTTPError as e:
                    print("Request returned a 404! Does the aritfact path exist?")
                    exit(-1)
                os.system("rpm -i {package}".format(package=package))
        else:
            for package in DEB_PACKAGES_UBUNTU:
                package = package.format(ext=suffix)
                full_url = URL+package
                print("Downloading from {URL}...".format(URL=full_url))
                try:
                    urllib.request.urlretrieve(full_url, package)
                except urllib.error.HTTPError as e:
                    print("Request returned a 404! Does the aritfact path exist?")
                    exit(-1)
                os.system("dpkg -i {package}".format(package=package))
