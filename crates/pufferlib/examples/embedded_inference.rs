//! Example of RL inference in a no_std environment.
//! Target: ESP32, RP2040, or other microcontrollers.

#![cfg(not(test))]
#![no_std]
#![no_main]

extern crate alloc;
use ndarray::ArrayD;
use pufferlib::prelude::*;

#[no_mangle]
pub extern "C" fn main() -> ! {
    // 1. Initialize memory allocator (implementation specific to hardware)
    // 2. Load policy (e.g. from flash memory)

    // 3. Inference loop
    loop {
        // Get observation from sensors
        // let obs = sensors.read();

        // Run inference
        // let action = policy.forward(&obs);

        // Act on hardware
        // actuators.apply(action);
    }
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
