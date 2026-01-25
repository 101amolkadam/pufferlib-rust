use super::SharedBuffer;
use std::ptr::NonNull;
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MapViewOfFile, UnmapViewOfFile, FILE_MAP_ALL_ACCESS, PAGE_READWRITE,
};

/// High-performance shared memory buffer using Win32 Named File Mappings.
pub struct Win32SharedBuffer {
    name: String,
    handle: HANDLE,
    ptr: NonNull<f32>,
    len: usize,
}

impl Win32SharedBuffer {
    /// Create a new named shared buffer
    pub fn new(name: &str, len: usize) -> Result<Self, String> {
        let size = (len * std::mem::size_of::<f32>()) as u32;
        
        // Convert name to UTF-16 for Win32 API
        let wide_name: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
        
        unsafe {
            let handle = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                size,
                wide_name.as_ptr(),
            );
            
            if handle.is_null() {
                return Err(format!("Failed to create file mapping: {}", name));
            }
            
            let ptr = MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, size as usize);
            if ptr.Value.is_null() {
                CloseHandle(handle);
                return Err(format!("Failed to map view of file: {}", name));
            }
            
            Ok(Self {
                name: name.to_string(),
                handle,
                ptr: NonNull::new_unchecked(ptr.Value as *mut f32),
                len,
            })
        }
    }
}

impl SharedBuffer for Win32SharedBuffer {
    fn as_ptr(&self) -> *mut f32 {
        self.ptr.as_ptr()
    }
    
    fn len(&self) -> usize {
        self.len
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for Win32SharedBuffer {
    fn drop(&mut self) {
        unsafe {
            UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: self.ptr.as_ptr() as *mut _ });
            CloseHandle(self.handle);
        }
    }
}

unsafe impl Send for Win32SharedBuffer {}
unsafe impl Sync for Win32SharedBuffer {}
