#!/usr/bin/env bash
set -euo pipefail

# Downloads a prebuilt wkhtmltopdf .deb, extracts the binary and places it
# into the app folder so it can be used without requiring apt install.
# Adjust URL if you need a different wkhtmltopdf version / distro.

OUT_DIR="/app/.wkhtml"
TMP_DEB="/tmp/wkhtml.deb"
TMP_EXTRACT="/tmp/wkhtml_extract"

if [ -x "${OUT_DIR}/wkhtmltopdf" ]; then
  echo "wkhtmltopdf already present at ${OUT_DIR}/wkhtmltopdf"
  exit 0
fi

mkdir -p "${OUT_DIR}"
rm -rf "${TMP_EXTRACT}"

# You may need to update this URL to a matching distro build. This one is
# from the wkhtmltopdf packaging releases (0.12.6-1 bionic amd64) and often
# works across Debian-derived images. Change if needed.
URL="https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox_0.12.6-1.bionic_amd64.deb"

echo "Downloading wkhtmltopdf from ${URL} ..."
curl -L --retry 3 -o "${TMP_DEB}" "${URL}"

echo "Extracting binary..."
mkdir -p "${TMP_EXTRACT}"
dpkg-deb -x "${TMP_DEB}" "${TMP_EXTRACT}"

if [ -f "${TMP_EXTRACT}/usr/bin/wkhtmltopdf" ]; then
  mv "${TMP_EXTRACT}/usr/bin/wkhtmltopdf" "${OUT_DIR}/wkhtmltopdf"
  chmod +x "${OUT_DIR}/wkhtmltopdf"
  echo "Installed wkhtmltopdf to ${OUT_DIR}/wkhtmltopdf"
else
  echo "ERROR: wkhtmltopdf binary not found inside the .deb"
  exit 2
fi

# cleanup
rm -f "${TMP_DEB}"
rm -rf "${TMP_EXTRACT}"

exit 0
