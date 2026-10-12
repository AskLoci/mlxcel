// Cargo build scripts compile for the host, so target_os cfgs cannot select
// the SDK or compiler runtime for an iOS cross-build.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AppleTarget {
    MacOs,
    IosDevice,
    IosSimulator,
}

impl AppleTarget {
    pub fn from_triple(target: &str) -> Option<Self> {
        match target {
            "aarch64-apple-darwin" | "x86_64-apple-darwin" => Some(Self::MacOs),
            "aarch64-apple-ios" => Some(Self::IosDevice),
            "aarch64-apple-ios-sim" | "x86_64-apple-ios" => Some(Self::IosSimulator),
            _ => None,
        }
    }

    pub fn sdk(self) -> &'static str {
        match self {
            Self::MacOs => "macosx",
            Self::IosDevice => "iphoneos",
            Self::IosSimulator => "iphonesimulator",
        }
    }

    pub fn compiler_runtime(self) -> &'static str {
        match self {
            Self::MacOs => "clang_rt.osx",
            Self::IosDevice => "clang_rt.ios",
            Self::IosSimulator => "clang_rt.iossim",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AppleTarget;

    #[test]
    fn cargo_target_selects_matching_sdk_and_runtime() {
        for (target, sdk, runtime) in [
            ("aarch64-apple-darwin", "macosx", "clang_rt.osx"),
            ("x86_64-apple-darwin", "macosx", "clang_rt.osx"),
            ("aarch64-apple-ios", "iphoneos", "clang_rt.ios"),
            (
                "aarch64-apple-ios-sim",
                "iphonesimulator",
                "clang_rt.iossim",
            ),
            ("x86_64-apple-ios", "iphonesimulator", "clang_rt.iossim"),
        ] {
            assert_eq!(
                AppleTarget::from_triple(target).map(AppleTarget::sdk),
                Some(sdk)
            );
            assert_eq!(
                AppleTarget::from_triple(target).map(AppleTarget::compiler_runtime),
                Some(runtime)
            );
        }
        assert_eq!(AppleTarget::from_triple("x86_64-unknown-linux-gnu"), None);
        assert_eq!(AppleTarget::from_triple("aarch64-apple-ios-macabi"), None);
    }
}
