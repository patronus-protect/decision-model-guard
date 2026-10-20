export default {
  async fetch(req, env) {
    const m = new URL(req.url).pathname.match(/^\/run\/(.+)$/);
    if (req.method !== "POST" || !m) return new Response("POST /run/<model>", { status: 404 });
    let body;
    try {
      body = await req.json();
    } catch (e) {
      return Response.json({ success: false, errors: [{ message: "the request body is not JSON" }] }, { status: 400 });
    }
    try {
      const result = await env.AI.run(decodeURIComponent(m[1]), body);
      return Response.json({ result, success: true });
    } catch (e) {
      const message = String((e && e.message) || e);
      // Workers AI input errors start with a 3xxx code ("3030: list parts needs items ..."): the caller must not retry them
      const status = /^3\d{3}\b/.test(message) ? 400 : 502;
      return Response.json({ success: false, errors: [{ message }] }, { status });
    }
  },
};
