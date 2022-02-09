// Calling the regl module with no arguments creates a full screen canvas and
// WebGL context, and then uses this context to initialize a new REGL instance
const regl = require('regl')()

import frag from './tri.frag'
import vert from './tri.vert'

// Calling regl() creates a new partially evaluated draw command
const drawTriangle = regl({
    // Shaders in regl are just strings.  You can use glslify or whatever you want
    // to define them.  No need to manually create shader objects.
    frag: frag,
    
    vert: vert,
    
    // Here we define the vertex attributes for the above shader
    attributes: {
        // regl.buffer creates a new array buffer object
        position: [
            [1/2,  -1/2],    // unrolls them into a typedarray (default Float32)
            [1/2,   1/2],
            [-1/2, -1/2],   // no need to flatten nested arrays, regl automatically
            [-1/2,  1/2]
        ]
        // regl automatically infers sane defaults for the vertex attribute pointers
    },
    
    primitive: 'triangle strip',

    uniforms: {
        // This defines the color of the triangle to be a dynamic variable
        color: regl.prop('color'),

        width: regl.context('viewportWidth'),
        height: regl.context('viewportHeight')
    },
  
    // This tells regl the number of vertices to draw in this command
    count: 4
})
    
// regl.frame() wraps requestAnimationFrame and also handles viewport changes
regl.frame(({time}) => {
    // clear contents of the drawing buffer
    regl.clear({
        color: [0, 0, 0, 1],
        depth: 1
    })
    
    // draw a triangle using the command defined above
    drawTriangle({
        color: [
            Math.cos(time * 0.001 * 1000) * 0.2 + 0.5,
            Math.sin(time * 0.0008 * 1000) * 0.2 + 0.5,
            Math.cos(time * 0.003 * 1000) * 0.2 + 0.5,
            1
        ]
    })
})