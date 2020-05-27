#![allow(clippy::cast_ptr_alignment)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::float_cmp)]

use async_std::task::block_on;
use rayon::prelude::*;
use std::future::Future;
use std::io::Cursor;
use std::ops::Deref;
use wgpu::*;
use zerocopy::AsBytes;

pub fn addition(left: &mut [f32], right: &[f32]) {
    for i in 0..left.len() {
        left[i] += right[i];
    }
}

pub unsafe fn addition_unchecked(left: &mut [f32], right: &[f32]) {
    for i in 0..left.len() {
        *left.get_unchecked_mut(i) += *right.get_unchecked(i);
    }
}

pub fn addition_iterator(left: &mut [f32], right: &[f32]) {
    left.iter_mut()
        .zip(right.iter())
        .for_each(|(left, right)| *left += *right);
}

pub fn addition_rayon(left: &mut [f32], right: &[f32]) {
    left.par_iter_mut()
        .zip(right.par_iter())
        .for_each(|(left, right)| *left += *right);
}

pub fn create_device() -> (Device, Queue) {
    let adapter = block_on(Adapter::request(
        &RequestAdapterOptions {
            power_preference: PowerPreference::Default,
            compatible_surface: None,
        },
        BackendBit::all(),
    ))
    .unwrap();

    block_on(adapter.request_device(&DeviceDescriptor {
        extensions: Extensions {
            anisotropic_filtering: false,
        },
        limits: Limits::default(),
    }))
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum UploadStyle {
    Mapping,
    Staging,
}

bitflags::bitflags! {
    pub struct AutomatedBufferUsage: u8 {
        const READ = 0b01;
        const WRITE = 0b10;
        const ALL = Self::READ.bits | Self::WRITE.bits;
    }
}

impl AutomatedBufferUsage {
    pub fn into_buffer_usage(self, style: UploadStyle) -> BufferUsage {
        let mut usage = BufferUsage::empty();
        if self.contains(Self::READ) {
            match style {
                UploadStyle::Mapping => usage.insert(BufferUsage::MAP_READ),
                UploadStyle::Staging => usage.insert(BufferUsage::COPY_SRC),
            }
        }
        if self.contains(Self::WRITE) {
            match style {
                UploadStyle::Mapping => usage.insert(BufferUsage::MAP_WRITE),
                UploadStyle::Staging => usage.insert(BufferUsage::COPY_DST),
            }
        }
        usage
    }
}

type BufferReadResult = Result<BufferReadMapping, BufferAsyncErr>;
type BufferWriteResult = Result<BufferWriteMapping, BufferAsyncErr>;

/// Represents either a mapping future (mapped style) or a function to create
/// a mapping future (buffered style).
enum ReadMapFn<MapFut, BufFunc> {
    Mapped(MapFut),
    Buffered(BufFunc),
}

impl<MapFut, BufFunc> ReadMapFn<MapFut, BufFunc>
where
    MapFut: Future<Output = BufferReadResult>,
    BufFunc: FnOnce() -> MapFut,
{
    /// Creates the mapping future
    fn prepare_future(self) -> MapFut {
        match self {
            ReadMapFn::Buffered(func) => func(),
            ReadMapFn::Mapped(mapped) => mapped,
        }
    }
}

/// A buffer which automatically uses either staging buffers or direct mapping to read/write to its
/// internal buffer based on the provided [`UploadStyle`]
pub struct AutomatedBuffer {
    inner: Buffer,
    style: UploadStyle,
    usage: AutomatedBufferUsage,
    size: BufferAddress,
}
impl AutomatedBuffer {
    /// Creates a new AutomatedBuffer with given settings. All operations directly
    /// done on the automated buffer according to `usage` will be added to the
    /// internal buffer's usage flags.
    pub fn new(
        device: &Device,
        size: BufferAddress,
        usage: AutomatedBufferUsage,
        other_usages: BufferUsage,
        label: Option<&str>,
        style: UploadStyle,
    ) -> Self {
        let inner = device.create_buffer(&BufferDescriptor {
            size,
            usage: usage.into_buffer_usage(style) | other_usages,
            label,
        });

        Self {
            inner,
            style,
            usage,
            size,
        }
    }

    /// Each of the two futures do different things based on the mapping style.
    ///
    /// Mapping:
    ///  - 1st: No-op
    ///  - 2nd: Resolves the mapping
    ///
    /// Buffered:
    ///  - 1st: Starts the staging buffer mapping
    ///  - 2nd: Resolves the mapping
    ///
    /// This is done with assistance of a generic helper type. The data for the first
    /// await is held by [`ReadMapFn`].
    fn map_read<MapFut, BufFunc>(
        mapping: ReadMapFn<MapFut, BufFunc>,
    ) -> impl Future<Output = impl Future<Output = BufferReadMapping>>
    where
        MapFut: Future<Output = BufferReadResult>,
        BufFunc: FnOnce() -> MapFut,
    {
        async move {
            // maps the staging buffer or passes forward the mapping of the real buffer
            let future = mapping.prepare_future();
            async move {
                // actually resolves the mapping
                future.await.unwrap()
            }
        }
    }

    /// Reads the underlying buffer using the proper read style.
    ///
    /// This function is unusual because it returns a future which itself returns a future. We shall
    /// refer to these as the First and the Second future.
    ///
    /// This function is safe, but has the following constraints so as to not cause a panic in wgpu:
    ///  - Buffer usage must contain [`READ`](AutomatedBufferUsage::READ).
    ///  - The buffer must not be in use by any other command buffer between calling this function
    ///    and calling await on the Second future.
    ///  - The First future must be awaited _after_ `encoder`'s command buffer is submitted to the queue and _before_ device.poll is called.
    ///  - The Second future mut be awaited _after_ device.poll is called and the mapping is resolved.
    ///
    /// Example:
    ///
    /// ```ignore
    /// let buffer = AutomatedBuffer::new(..);
    ///
    /// let map_read_buf1 = buffer.read_from_buffer(&device, &mut encoder);
    /// queue.submit(&[encoder.submit()]); // must happen before first await
    ///
    /// let map_read_buf2 = map_read_buf1.await;
    /// device.poll(...); // must happen before second await and after first
    ///
    /// let mapping = map_read_buf2.await;
    /// // use mapping
    /// ```
    pub fn read_from_buffer(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
    ) -> impl Future<Output = impl Future<Output = BufferReadMapping>> {
        assert!(
            self.usage.contains(AutomatedBufferUsage::READ),
            "Must have usage READ to read from buffer. Current usage {:?}",
            self.usage
        );
        match self.style {
            UploadStyle::Mapping => {
                Self::map_read(ReadMapFn::Mapped(self.inner.map_read(0, self.size)))
            }
            UploadStyle::Staging => {
                let staging = device.create_buffer(&BufferDescriptor {
                    size: self.size,
                    usage: BufferUsage::MAP_READ | BufferUsage::COPY_DST,
                    label: Some("read dst buffer"),
                });
                encoder.copy_buffer_to_buffer(&self.inner, 0, &staging, 0, self.size);
                let size = self.size;
                Self::map_read(ReadMapFn::Buffered(move || staging.map_read(0, size)))
            }
        }
    }

    /// When the returned future is awaited, writes the data to the buffer if it is a mapped buffer.
    /// No-op for the use of a staging buffer.
    fn map_write<'a>(
        data: &'a [u8],
        mapping: Option<impl Future<Output = BufferWriteResult> + 'a>,
    ) -> impl Future<Output = ()> + 'a {
        async move {
            if let Some(mapping) = mapping {
                mapping.await.unwrap().as_slice().copy_from_slice(data);
            }
        }
    }

    /// Writes to the underlying buffer using the proper write style.
    ///
    /// This function is safe, but has the following constraints so as to not cause a panic in wgpu:
    ///  - Buffer usage must contain [`WRITE`](AutomatedBufferUsage::WRITE)
    ///  - The returned future must be awaited _after_ calling device.poll() to resolve it.
    ///  - The command buffer created by `encoder` must **not** be submitted to a queue before this future is awaited.
    ///
    /// Example:
    ///
    /// ```ignore
    /// let buffer = AutomatedBuffer::new(..);
    ///
    /// let map_write = buffer.write_to_buffer(&device, &mut encoder, &data);
    /// device.poll(...); // must happen before await
    ///
    /// let mapping = map_write.await; // Calling await will write to the mapping
    ///
    /// queue.submit(&[encoder.submit()]); // must happen after await
    /// ```
    pub fn write_to_buffer<'a>(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        data: &'a [u8],
    ) -> impl Future<Output = ()> + 'a {
        assert!(
            self.usage.contains(AutomatedBufferUsage::WRITE),
            "Must have usage WRITE to write to buffer. Current usage {:?}",
            self.usage
        );
        match self.style {
            UploadStyle::Mapping => Self::map_write(
                data,
                Some(self.inner.map_write(0, data.len() as BufferAddress)),
            ),
            UploadStyle::Staging => {
                let staging = device.create_buffer_with_data(data, BufferUsage::COPY_SRC);
                encoder.copy_buffer_to_buffer(
                    &staging,
                    0,
                    &self.inner,
                    0,
                    data.len() as BufferAddress,
                );
                Self::map_write(data, None)
            }
        }
    }
}

impl Deref for AutomatedBuffer {
    type Target = Buffer;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

pub struct GPUAddition<'a> {
    device: &'a Device,
    queue: &'a Queue,
    pipeline: ComputePipeline,
    left_buffer: AutomatedBuffer,
    right_buffer: AutomatedBuffer,
    bind_group: BindGroup,
    commands: Vec<CommandBuffer>,
}

impl<'a> GPUAddition<'a> {
    pub async fn new(
        device: &'a Device,
        queue: &'a Queue,
        style: UploadStyle,
        size: usize,
    ) -> GPUAddition<'a> {
        let size_bytes = size as BufferAddress * 4;

        let shader_source = include_bytes!("addition.spv");
        let shader_module =
            device.create_shader_module(&read_spirv(Cursor::new(&shader_source[..])).unwrap());

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            bindings: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: false,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: true,
                    },
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::UniformBuffer { dynamic: false },
                },
            ],
            label: Some("bind group layout"),
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            layout: &pipeline_layout,
            compute_stage: ProgrammableStageDescriptor {
                module: &shader_module,
                entry_point: "main",
            },
        });

        let left_buffer = AutomatedBuffer::new(
            &device,
            size_bytes,
            AutomatedBufferUsage::ALL,
            BufferUsage::STORAGE,
            Some("left buffer"),
            style,
        );

        let right_buffer = AutomatedBuffer::new(
            &device,
            size_bytes,
            AutomatedBufferUsage::WRITE,
            BufferUsage::STORAGE_READ,
            Some("right buffer"),
            style,
        );

        let uniform_buffer =
            device.create_buffer_with_data((size as u32).as_bytes(), BufferUsage::UNIFORM);

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &bind_group_layout,
            bindings: &[
                Binding {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: &left_buffer,
                        range: 0..size_bytes,
                    },
                },
                Binding {
                    binding: 1,
                    resource: BindingResource::Buffer {
                        buffer: &right_buffer,
                        range: 0..size_bytes,
                    },
                },
                Binding {
                    binding: 2,
                    resource: BindingResource::Buffer {
                        buffer: &uniform_buffer,
                        range: 0..4,
                    },
                },
            ],
            label: Some("bind group"),
        });

        Self {
            device,
            queue,
            pipeline,
            left_buffer,
            right_buffer,
            bind_group,
            commands: Vec::default(),
        }
    }

    pub async fn set_buffers(&mut self, left: &[f32], right: &[f32]) {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("set buffers"),
            });

        let map_left =
            self.left_buffer
                .write_to_buffer(&self.device, &mut encoder, left.as_bytes());
        let map_right =
            self.right_buffer
                .write_to_buffer(&self.device, &mut encoder, right.as_bytes());

        self.device.poll(Maintain::Wait);

        map_left.await;
        map_right.await;

        // Ensure copy takes place during run
        self.commands.push(encoder.finish());
    }

    pub async fn run(&mut self, size: usize) -> BufferReadMapping {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("compute encoder"),
            });
        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch((size as u32 + 63) / 64, 1, 1);
        drop(cpass);

        self.commands.push(encoder.finish());
        self.queue.submit(&self.commands);
        self.commands.clear();

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("read mapping encoder"),
            });

        let map_left = self
            .left_buffer
            .read_from_buffer(&self.device, &mut encoder);
        self.queue.submit(&[encoder.finish()]);
        let map_left = map_left.await;
        self.device.poll(Maintain::Wait);
        map_left.await
    }
}

#[cfg(test)]
mod test {
    use crate::{create_device, GPUAddition, UploadStyle};
    use async_std::task::block_on;
    use itertools::{zip, Itertools};

    macro_rules! addition_test {
        ($name:ident, $function:expr) => {
            #[test]
            fn $name() {
                let mut left = (0..10000).map(|v| v as f32).collect_vec();
                let right = (0..10000).map(|v| (v + 1) as f32).collect_vec();
                let result = (0..10000).map(|v| (v + v + 1) as f32).collect_vec();

                $function(&mut left, &right);

                for (i, (left, result)) in zip(left, result).enumerate() {
                    assert_eq!(left, result, "Index {} failed", i);
                }
            }
        };
    }

    addition_test!(addition, |left: &mut [f32], right: &[f32]| {
        crate::addition(left, right);
    });
    addition_test!(addition_unchecked, |left: &mut [f32], right: &[f32]| {
        unsafe { crate::addition_unchecked(left, right) };
    });
    addition_test!(addition_iterator, |left: &mut [f32], right: &[f32]| {
        crate::addition_iterator(left, right);
    });
    addition_test!(addition_rayon, |left: &mut [f32], right: &[f32]| {
        crate::addition_rayon(left, right);
    });
    addition_test!(addition_gpu_mapping, |left: &mut [f32], right: &[f32]| {
        let (device, queue) = create_device();
        let mut gpu = block_on(GPUAddition::new(
            &device,
            &queue,
            UploadStyle::Mapping,
            left.len(),
        ));
        block_on(gpu.set_buffers(&left, &right));
        let result_mapping = block_on(gpu.run(left.len()));
        let bytes = result_mapping.as_slice();
        let floats =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) };
        assert_eq!(left.len(), floats.len());
        left.copy_from_slice(floats);
    });
    addition_test!(addition_gpu_staging, |left: &mut [f32], right: &[f32]| {
        let (device, queue) = create_device();
        let mut gpu = block_on(GPUAddition::new(
            &device,
            &queue,
            UploadStyle::Staging,
            left.len(),
        ));
        block_on(gpu.set_buffers(&left, &right));
        let result_mapping = block_on(gpu.run(left.len()));
        let bytes = result_mapping.as_slice();
        let floats =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) };
        assert_eq!(left.len(), floats.len());
        left.copy_from_slice(floats);
    });
}
