precision mediump float;

attribute vec2 position;

uniform float width, height;

void main() {
    vec2 s = min(width, height) / vec2(width, height);
    gl_Position = vec4(position * s, 0, 1);
}