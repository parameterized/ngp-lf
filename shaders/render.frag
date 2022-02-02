
precision mediump float;
//precision highp int;

uniform sampler2D viewTex0;
uniform sampler2D viewTex1;
uniform vec3 viewPos0;
uniform vec3 viewPos1;
uniform mat4 viewMatInv0;
uniform mat4 viewMatInv1;

const int GRID_LEVELS = 8;
//const int TABLE_SIZE = 6400;

uniform float fovFactor;
uniform vec3 camPos;
uniform sampler2D hashTable;

varying vec3 vertPos;


// https://gist.github.com/mattatz/70b96f8c57d4ba1ad2cd
/*
const int BIT_COUNT = 16;

int modi(int x, int y) {
    return x - y * (x / y);
}

int or(int a, int b) {
    int result = 0;
    int n = 1;

    for(int i = 0; i < BIT_COUNT; i++) {
        if ((modi(a, 2) == 1) || (modi(b, 2) == 1)) {
            result += n;
        }
        a = a / 2;
        b = b / 2;
        n = n * 2;
        if(!(a > 0 || b > 0)) {
            break;
        }
    }
    return result;
}

int and(int a, int b) {
    int result = 0;
    int n = 1;

    for(int i = 0; i < BIT_COUNT; i++) {
        if ((modi(a, 2) == 1) && (modi(b, 2) == 1)) {
            result += n;
        }

        a = a / 2;
        b = b / 2;
        n = n * 2;

        if(!(a > 0 && b > 0)) {
            break;
        }
    }
    return result;
}

int inot(int a) {
    int result = 0;
    int n = 1;
    
    for(int i = 0; i < BIT_COUNT; i++) {
        if (modi(a, 2) == 0) {
            result += n;    
        }
        a = a / 2;
        n = n * 2;
    }
    return result;
}

int xor(int a, int b) {
    return and(or(a, b), or(inot(a), inot(b)));
}
*/


// https://www.shadertoy.com/view/4djSRW
vec4 hash44(vec4 p4) {
	p4 = fract(p4 * vec4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy + 33.33);
    return fract((p4.xxyz + p4.yzzw) * p4.zywx);
}

/*
int pi1 = 183761437; // 1 not valid when using hash index directly as color value
int pi2 = 2654435761;
int pi3 = 805459861;
*/

vec4 sample(vec3 p, int level) {
    float hash = hash44(floor(vec4(p, float(level)))).x;
    return texture2D(hashTable, vec2(hash, 0.5));
}

vec4 sample_bilinear(vec3 p, int level) {
    p *= pow(2.0, float(level));
    vec2 o = vec2(0.0, 1.0);
    vec4 c000 = sample(p, level);
    vec4 c001 = sample(p + o.xxy, level);
    vec4 c010 = sample(p + o.xyx, level);
    vec4 c011 = sample(p + o.xyy, level);
    vec4 c100 = sample(p + o.yxx, level);
    vec4 c101 = sample(p + o.yxy, level);
    vec4 c110 = sample(p + o.yyx, level);
    vec4 c111 = sample(p + o.yyy, level);
    
    vec4 c00 = mix(c000, c001, fract(p.z));
    vec4 c01 = mix(c010, c011, fract(p.z));
    vec4 c10 = mix(c100, c101, fract(p.z));
    vec4 c11 = mix(c110, c111, fract(p.z));

    vec4 c0 = mix(c00, c01, fract(p.y));
    vec4 c1 = mix(c10, c11, fract(p.y));

    return mix(c0, c1, fract(p.x));
}

void main() {
    /*
    vec3 camToPixel0 = vertPos - viewPos0;
    vec4 localRd0 = viewMatInv0 * vec4(camToPixel0, 0.0);
    // aspect ratio is 1
    vec4 tex0 = texture2D(viewTex0, (localRd0.xy / -localRd0.z * fovFactor + 1.0) / 2.0);

    vec3 camToPixel1 = vertPos - viewPos1;
    vec4 localRd1 = viewMatInv1 * vec4(camToPixel1, 0.0);
    vec4 tex1 = texture2D(viewTex1, (localRd1.xy / -localRd1.z * fovFactor + 1.0) / 2.0);

    float t = distance(camPos, viewPos0) / (distance(camPos, viewPos0) + distance(camPos, viewPos1));
    gl_FragColor = mix(tex0, tex1, t);
    */

    vec3 rd = normalize(vertPos * 5.0 - camPos);

    // localize rays by using closest point to object as origin
    // (object at 0,0,0)
    float t = dot(-camPos, rd);
    vec3 localOrigins = camPos + rd * t;

    // test on surface
    //localOrigins = vertPos * 4.99;


    localOrigins = (localOrigins / 5.0 + 0.5);// * float(gridSize);
    int x = int(localOrigins.x);
    int y = int(localOrigins.y);
    int z = int(localOrigins.z);
    // logical int ops not supported in glsl 1.0
    //int hash = modi(xor(xor(x * pi1, y * pi2), z * pi3), tableSize);
    //int hash = modi(x * pi1 + y * pi2 + z * pi3, tableSize);

    vec4 c = vec4(1.0);
    for (int i = 0; i < GRID_LEVELS; i++) {
        c *= sample_bilinear(localOrigins, i + 1); 
    }
    //c /= float(GRID_LEVELS);
    c /= pow(0.5, 7.5);

    gl_FragColor = c;
    //gl_FragColor = vec4(vec3(length(vertPos * 5.0 - camPos)), 1.0);
    //gl_FragColor = vec4(localOrigins, 1.0);
}
