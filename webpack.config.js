
const path = require('path');

module.exports = function (env, argv) {
    let prod = env && env.production;
    return {
        mode: prod ? 'production' : 'development',
        entry: {
            main: './src/index.js',
            regltest: './src/regltest/index.js'
        },
        output: {
            path: path.join(__dirname, 'public'),
            filename: `[name]-bundle.js`
        },
        module: {
            rules: [
                {
                    test: /\.js$/,
                    exclude: /node_modules/,
                    loader: 'babel-loader'
                },
                {
                    test: /\.(glsl|vs|fs|vert|frag)$/,
                    exclude: /node_modules/,
                    use: [
                        'raw-loader',
                        'glslify-loader'
                    ]
                }
            ]
        },
        devtool: 'eval-cheap-source-map',
        devServer: {
            static: {
                directory: path.join(__dirname, 'public')
            },
            compress: true,
            port: 8080,
            historyApiFallback: {
                rewrites: [
                    { from: /^\/regltest/, to: 'regltest.html' }
                ]
            }
        }
    };
};