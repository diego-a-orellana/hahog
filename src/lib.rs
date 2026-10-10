#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]
#![warn(missing_docs, clippy::missing_docs_in_private_items)]

mod algorithm;
mod api;
mod error;

pub use api::Hahog;
pub use error::{Error, Result};
