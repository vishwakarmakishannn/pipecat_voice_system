# React + Vite

## Production voice ingress

The bundled Nginx container terminates plain HTTP inside the deployment and
proxies signaling/API traffic to the backend over a reusable upstream socket.
Production TLS must terminate at a colocated ingress/load balancer. Configure
that ingress with TLS session resumption and HTTP/2 or HTTP/3, preserve
`X-Forwarded-Proto`, and disable response/request buffering for `/start` and
`/api/offer`. Avoid inserting additional cross-region proxies in the media
signaling path. The bundled upstream connect timeout is 2 seconds; read timeout
is 30 seconds for signaling/control requests.

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
