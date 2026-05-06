#!/usr/bin/env bash
# Scan the local environment for indicators of compromise from the
# March 24 2026 LiteLLM supply-chain attack (TeamPCP).
#
# Reference:
#   https://docs.litellm.ai/blog/security-update-march-2026
#   https://huggingface.co/blog/davidberenstein1957/litellm-supply-chain-attack-2026
#
# Compromised versions: 1.82.7 and 1.82.8 only.
# Latest known-clean: >= 1.83.0 (released Mar 30 2026 from LiteLLM CI/CD v2).
#
# This script returns non-zero if any IOC is found.

set -u

red()   { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }

failures=0
report_fail() { red "FAIL: $*"; failures=$((failures+1)); }
report_ok()   { green "OK:   $*"; }

# 1. Check installed litellm version (if any)
if command -v python3 >/dev/null 2>&1; then
    ver=$(python3 -c "import importlib.metadata as m; print(m.version('litellm'))" 2>/dev/null || echo "")
    if [ -z "$ver" ]; then
        yellow "INFO: litellm not installed in the active Python interpreter"
    else
        case "$ver" in
            1.82.7|1.82.8)
                report_fail "Installed litellm==$ver is COMPROMISED. Uninstall and rotate credentials."
                ;;
            1.82.6|1.82.5|1.82.4|1.82.3|1.82.2|1.82.1|1.82.0|1.81.*|1.80.*|1.79.*|1.78.*)
                yellow "INFO: litellm==$ver is verified clean but predates the post-incident hardening."
                yellow "      Recommend upgrading to >=1.83.0."
                ;;
            *)
                # Crude semver compare: anything starting with 1.83., 1.84., ... is fine
                case "$ver" in
                    1.83.*|1.84.*|1.85.*|1.86.*|1.87.*|1.88.*|1.89.*|1.9*|2.*)
                        report_ok "litellm==$ver (post-incident hardened release)"
                        ;;
                    *)
                        yellow "INFO: litellm==$ver - manually verify against https://docs.litellm.ai/blog/security-update-march-2026"
                        ;;
                esac
                ;;
        esac
    fi
fi

# 2. Scan installed packages for the malware launcher .pth file
mapfile -t site_dirs < <(python3 -c "import site; print('\n'.join(site.getsitepackages() + [site.getusersitepackages()]))" 2>/dev/null || true)
for d in "${site_dirs[@]}"; do
    [ -d "$d" ] || continue
    if find "$d" -maxdepth 2 -name "litellm_init.pth" -print 2>/dev/null | grep -q .; then
        report_fail "Found litellm_init.pth under $d (IOC for v1.82.8)"
    fi
done

# 3. Scan home dir for the persistence backdoor
for path in \
    "$HOME/.config/sysmon" \
    "$HOME/.config/sysmon/sysmon.py" \
    "$HOME/.config/systemd/user/sysmon.service" ; do
    if [ -e "$path" ]; then
        report_fail "Found persistence artifact: $path"
    fi
done

# 4. Look for outbound DNS references in any cached litellm installs
if [ -d "$HOME/.cache/uv" ] || [ -d "$HOME/.cache/pip" ]; then
    if grep -RIl --include='*.py' --include='*.whl' \
        -e 'models.litellm.cloud' -e 'checkmarx.zone' \
        "$HOME/.cache/uv" "$HOME/.cache/pip" 2>/dev/null | grep -q .; then
        report_fail "Found IOC domain reference in package cache (models.litellm.cloud / checkmarx.zone)"
    fi
fi

if [ "$failures" -eq 0 ]; then
    green "All clear: no LiteLLM IOCs detected."
    exit 0
else
    red "$failures IOC check(s) failed. Follow the remediation steps in"
    red "https://docs.litellm.ai/blog/security-update-march-2026"
    exit 1
fi
