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

DEB_PACKAGES_CENTOS=[
"libnvinfer8-8.5.1-1.cuda{ver}.x86_64.rpm",
"libnvonnxparsers8-8.5.1-1.cuda{ver}.x86_64.rpm",
"libnvparsers8-8.5.1-1.cuda{ver}.x86_64.rpm",
"libnvinfer-plugin8-8.5.1-1.cuda{ver}.x86_64.rpm",
"libnvinfer-devel-8.5.1-1.cuda{ver}.x86_64.rpm",
"libnvparsers-devel-8.5.1-1.cuda{ver}.x86_64.rpm",
"libnvonnxparsers-devel-8.5.1-1.cuda{ver}.x86_64.rpm",
"libnvinfer-plugin-devel-8.5.1-1.cuda{ver}.x86_64.rpm",
"python3-libnvinfer-8.5.1-1.cuda{ver}.x86_64.rpm",
]

DEB_PACKAGES_UBUNTU=[
    "libnvinfer8_8.5.1-1{ext}",
    "libnvonnxparsers8_8.5.1-1{ext}",
    "libnvparsers8_8.5.1-1{ext}",
    "libnvinfer-plugin8_8.5.1-1{ext}",
    "libnvinfer-dev_8.5.1-1{ext}",
    "libnvonnxparsers-dev_8.5.1-1{ext}",
    "libnvparsers-dev_8.5.1-1{ext}",
    "libnvinfer-plugin-dev_8.5.1-1{ext}",
    "python3-libnvinfer_8.5.1-1{ext}",
]

DEB_PACKAGES_CROSS_SBSA=[
    "libnvinfer8-cross-sbsa_8.5.1-1+cuda11.8_all.deb",
    "libnvinfer-dev-cross-sbsa_8.5.1-1+cuda11.8_all.deb",
    "libnvinfer-plugin-dev-cross-sbsa_8.5.1-1+cuda11.8_all.deb",
    "libnvinfer-plugin8-cross-sbsa_8.5.1-1+cuda11.8_all.deb",
    "libnvonnxparsers-dev-cross-sbsa_8.5.1-1+cuda11.8_all.deb",
    "libnvonnxparsers8-cross-sbsa_8.5.1-1+cuda11.8_all.deb",
    "libnvparsers-dev-cross-sbsa_8.5.1-1+cuda11.8_all.deb",
    "libnvparsers8-cross-sbsa_8.5.1-1+cuda11.8_all.deb",
]

ROOT_URL = "http://cuda-repo/release-candidates/Libraries/TensorRT/v8.5/8.5.1.1-78a84d23/"

def get_cuda_props(cuda_ver):
    assert len(cuda_ver) >= 4
    cuda = cuda_ver[:4]
    if (cuda == "11.8"):
        return cuda, cuda+"-r520"
    elif (cuda == "10.2"):
        return cuda, cuda+"-r440"
    else:
        print("Found unsupported CUDA version: " + cuda)
        exit(-1)

def get_arch_props(os):
    if os == "22.04" or os == "20.04" or os == "18.04":
        return "amd64", "Ubuntu{ver}-x64-agnostic".format(ver=os.replace(".", "_")), "deb", "+", "_", False
    elif os == "7":
        return "x86_64", "RHEL7_9-x64-agnostic", "rpm", ".", ".", True
    elif os == "8":
        return "x86_64", "RHEL8_3-x64-agnostic", "rpm", ".", ".", True
    elif os == "cross-sbsa":
        return "all", "Ubuntu20_04-aarch64", "deb", ".", ".", False
    else:
        print("Found unsupported OS: " + os)
        exit(-1)

if __name__ == "__main__":
    args = parse_arguments()
    cuda, cuda_url  = get_cuda_props(args.cuda)
    arch, url, ext, prefix, sep, is_centos = get_arch_props(args.os)
    suffix = "{prefix}cuda{cuda}{sep}{arch}.{ext}".format(prefix=prefix, cuda=cuda, sep=sep, arch=arch, ext=ext)

    with tempfile.TemporaryDirectory() as tmp:
        URL = ROOT_URL + "{cuda}/{url}/{ext}/".format(cuda=cuda_url, url=url, ext=ext)

        if args.os == "cross-sbsa":
            for package in DEB_PACKAGES_CROSS_SBSA:
                full_url = URL+package
                print("Downloading from {URL}...".format(URL=full_url))
                try:
                    urllib.request.urlretrieve(full_url, package)
                except urllib.error.HTTPError as e:
                    print("Request returned a 404! Does the aritfact path exist?")
                    exit(-1)
                os.system("dpkg -i {package}".format(package=package))
        elif is_centos:
            for package in DEB_PACKAGES_CENTOS:
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
