# ngp-lf
[Neural graphics primitives](https://nvlabs.github.io/instant-ngp/) hash encoding applied to light fields, implemented in WebGL shaders

Training is not implemented yet

## Running

The lastest version will be hosted on Github [here](https://parameterized.github.io/ngp-lf).

If you want to modify the code, install the dependencies (`npm install`) then `npm run dev-server` to start the webpack development server.

## Controls
- Right click and hold + Mouse / WASD / Space to move camera
- Hold Shift to move faster
- Tab to cycle through view modes (predicted view, closest view)
- 1 to set your camera to a dataset view
- O to toggle automatic camera orbiting

## Attribution

- The lego data is from [NeRF](https://github.com/bmild/nerf), using [Heinzelnisse's model](https://www.blendswap.com/blend/11490)
- Currently uses [David Hoskins' hash function](https://www.shadertoy.com/view/4djSRW)