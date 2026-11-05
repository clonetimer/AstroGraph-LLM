# AstroGraph Portable V2.6.2 Preview — CHECK.bat Windows path fix

## Fixed

The V2.6 and V2.6.1 `CHECK.bat` commands explicitly passed `"%~dp0"` as the final Python argument. Windows expands `%~dp0` with a trailing backslash, which the Python/Windows C-runtime argument parser can interpret as escaping the closing double quote. This produced malformed paths such as `llm" \MANIFEST.json` and the misleading `OSError: Invalid argument` message.

The verifier already detects the package root from its own script path, so **the extraneous root argument was removed**. Neither the checksum verifier nor its SHA-256 validation was weakened. The root and `source/portable/` launcher copies were both updated.

## Verification

- Reproduced the old command-line failure using Windows C-runtime quoting rules with the user-supplied path, plus a space/Chinese-character path.
- The fixed command produces exactly 5 Python argv entries: executable, `-X`, `utf8`, `-B`, script path. No directory argument.
- The source test verifies that the manifest checker works from a different working directory without a root argument, and that tampering is rejected.
- Full bundle ZIP CRC, SHA-256 manifest, and portable test suite are re-run on the new archive.
- **This is a Linux test environment; native Windows 10 execution remains unverified.**

## Existing V2.6 / V2.6.1 installations

Use the separately supplied `CHECK_FIXED.bat` beside `CHECK.bat` and `MANIFEST.json`. Do **not** overwrite the original tracked `CHECK.bat`, since the old manifest records its previous hash. The standalone script validates the original files and invokes the original self-test without requiring package modifications.

## Limitations

The package still does not include a model runtime/weights. It does not establish Windows 10 Stable acceptance or production LLM quality. No application analysis behavior changed.
