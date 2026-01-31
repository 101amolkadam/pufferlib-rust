//! Example of RL inference in a no_std environment.
//! Target: ESP32, RP2040, or other microcontrollers.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(not(feature = "std"), no_main)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
#[no_mangle]
pub extern "C" fn main() -> ! {
    loop {}
}

#[cfg(not(feature = "std"))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[cfg(feature = "std")]
fn main() {
    println!("This example is for no_std environments.");
}
