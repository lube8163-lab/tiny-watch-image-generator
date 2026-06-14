#!/bin/sh
set -eu

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="SDXLDesktopCoreMLTest.app"
DESTINATION="${HOME}/Desktop/${APP_NAME}"

xcodebuild \
  -project "${PROJECT_ROOT}/SDXLCoreMLTest.xcodeproj" \
  -scheme SDXLDesktopCoreMLTest \
  -destination "platform=macOS,variant=Mac Catalyst" \
  CODE_SIGNING_ALLOWED=NO \
  build

APP_PATH="$(find "${HOME}/Library/Developer/Xcode/DerivedData" \
  -path "*/Build/Products/Debug-maccatalyst/${APP_NAME}" \
  -type d \
  -print \
  | head -n 1)"

if [ -z "$APP_PATH" ]; then
  echo "Built app was not found in DerivedData." >&2
  exit 1
fi

rm -rf "$DESTINATION"
/usr/bin/ditto --noextattr --noqtn "$APP_PATH" "$DESTINATION"
xattr -cr "$DESTINATION"

echo "Installed: $DESTINATION"
