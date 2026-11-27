#!/usr/bin/env bash
# Build the LML (Laboratoire de Mecanique de Lille) synthetic PIV image generator suite
#   PGlml  : particle parameter generator (x, y, sigma, level)
#   DGlml  : particle displacement generator (translation / "plasma"; single or double exposure)
#   SIGlml : synthetic image generator (Gaussian particles + noise + background ramp)
# by S. Coudert (LML), https://github.com/coupdair/{PGlml,DGlml,SIGlml}, using CImg 1.3.0
# (archived by the same author in https://github.com/coupdair/CImg.archive).
#
# Two local build fixes, sources are not modified (patched copies *_nodbg.cpp are compiled):
#  1. cimg_debug 2 -> 0 : debug display() calls abort without an X display (DGlml plasma mode
#     calls display() before writing its output).
#  2. SIGlml calls add_sprite(sprite,x,y,z,v,mode) but the published plugin
#     (CImg.PlugIn/PlugIn.CImg.sprite_multi_layer.h) only has add_sprite(sprite,x,y,z,v)
#     (additive).  A shim forwards the 6-argument call to it (mode ignored -> additive).
#
# usage: scripts/build_lml.sh [dir]      (default dir: third_party/lml, git-ignored)
set -euo pipefail
D=${1:-third_party/lml}
mkdir -p "$D" && cd "$D"
for r in PGlml DGlml SIGlml CImg.PlugIn CImg.archive; do
  [ -d $r ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 https://github.com/coupdair/$r $r
done
mkdir -p CImg build
python3 - <<'EOF'
import zipfile
open("CImg/CImg.h", "wb").write(zipfile.ZipFile("CImg.archive/CImg-1.3.0.zip").read("CImg-1.3.0/CImg.h"))
EOF
cat > build/PlugIn.shim.h <<'EOF'
#include "../CImg.PlugIn/PlugIn.CImg.sprite_multi_layer.h"
template<typename t>
CImg<T>& add_sprite(const CImg<t>& sprite,const int x0,const int y0,const int z0,const int v0,const int /*mode*/)
{ return add_sprite(sprite,x0,y0,z0,v0); }
EOF
F='-O2 -std=gnu++98 -fpermissive -w -Dcimg_display=0 -DVERSION="v" -DPG_FORMAT_VERSION="v0.0.1:xySB"'
for t in PGlml DGlml SIGlml; do
  sed -e 's/#define cimg_debug 2/#define cimg_debug 0/' \
      -e 's#"../CImg.PlugIn/PlugIn.CImg.sprite_multi_layer.h"#"../build/PlugIn.shim.h"#' \
      $t/$t.cpp > $t/${t}_nodbg.cpp
  (cd $t && g++ $F -o $t ${t}_nodbg.cpp -lpthread)
done
echo "built: $(pwd)/{PGlml,DGlml,SIGlml}"
