// Copyright (c) 2026 ARC (Applied Research & Computation)
// SPDX-License-Identifier: LGPL-2.1-or-later

#[repr(C)]
pub struct ArrayHandle {
    _private: [u8; 0],
}

unsafe extern "C" {
    pub fn array_create(data: *const f64, len: usize) -> *mut ArrayHandle;
    pub fn array_destroy(arr: *mut ArrayHandle);
    pub fn array_add(result: *mut ArrayHandle, a: *const ArrayHandle, b: *const ArrayHandle);
    pub fn array_add_backward(
        res: *mut ArrayHandle, dres: *mut ArrayHandle,
        a: *const ArrayHandle, da: *mut ArrayHandle,
        b: *const ArrayHandle, db: *mut ArrayHandle
    );
    pub fn array_add_forward(
        res: *mut ArrayHandle, dres: *mut ArrayHandle,
        a: *const ArrayHandle, da: *mut ArrayHandle,
        b: *const ArrayHandle, db: *mut ArrayHandle
    );
    pub fn array_copy(arr: *const ArrayHandle, buffer: *mut f64, len: usize);
}

pub struct CppArray {
    handle: *mut ArrayHandle,
    len: usize,
}

impl CppArray {
    pub fn new(data: Vec<f64>) -> Self {
        let len = data.len();
        let handle = unsafe { array_create(data.as_ptr(), len) };
        Self { handle, len }
    }

    pub fn add(&self, other: &Self) -> Result<Self, String> {
        let result = CppArray::new(vec![0.0; self.len]);
        unsafe { array_add(result.handle, self.handle, other.handle) };
        Ok(result)
    }

    pub fn add_backward(a: &Self, b: &Self, dres: &Self) -> (Self, Self) {
        let res = CppArray::new(vec![0.0; a.len]);
        let da = CppArray::new(vec![0.0; a.len]);
        let db = CppArray::new(vec![0.0; a.len]);

        unsafe { 
            array_add_backward(
                res.handle, dres.handle,
                a.handle, da.handle,
                b.handle, db.handle
            );
        }

        (da, db)
    }

    pub fn add_forward(a: &Self, da: &Self, b: &Self, db: &Self) -> Self {
        let res = CppArray::new(vec![0.0; a.len]);
        let dres = CppArray::new(vec![0.0; a.len]);

        unsafe { 
            array_add_forward(
                res.handle, dres.handle,
                a.handle, da.handle,
                b.handle, db.handle
            );
        }

        dres
    }

    pub fn to_vec(&self) -> Vec<f64> {
        let mut buffer = vec![0.0; self.len];
        unsafe { array_copy(self.handle, buffer.as_mut_ptr(), self.len ) };
        buffer
    }
}

impl Drop for CppArray {
    fn drop(&mut self) {
        unsafe { array_destroy(self.handle); }
    }
}