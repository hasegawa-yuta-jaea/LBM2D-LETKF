#!/bin/bash -ue

code=EigenG-Batched
version=1.2
url=https://www.r-ccs.riken.jp/labs/lpnctrt/projects/eigengbatched/$code-$version.tar.gz

(cd src; curl -L $url | tar zvxf -)

srcdir=src/$code-$version
make -C $srcdir

cat > config/makefile.$code.in << EOT
### EigenG
CXXFLAGS += -I $srcdir -D USE_EIGENG_BATCHED
LDFLAGS += -L $srcdir -l eigenGbatch
EOT
