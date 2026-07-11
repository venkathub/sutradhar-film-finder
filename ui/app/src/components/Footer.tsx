// Attribution chrome (P6 task 5) — the LICENSING.md obligations as RENDERED
// UI, present on every screen:
// - TMDB: the official logo identifies the API use; the exact FAQ notice; the
//   logo renders LESS PROMINENT than the Sutradhar mark (an explicit TMDB
//   condition — enforced by a size assertion in Footer.test.tsx);
// - Wikipedia: visible CC BY-SA 4.0 label (per-claim links are additionally
//   revision-pinned via ?oldid= — see lib/citations.ts);
// - IMDb: AKA/dub titles derive from title.akas → the courtesy line + the
//   non-commercial-demo note (IMDb non-commercial dataset terms).
import tmdbLogo from "../assets/tmdb.svg";

export const TMDB_NOTICE =
  "This product uses the TMDB API but is not endorsed or certified by TMDB.";

export const IMDB_COURTESY =
  "Information courtesy of IMDb (https://www.imdb.com). Used with permission.";

export default function Footer() {
  return (
    <footer className="attribution" data-testid="attribution-footer">
      <p className="attribution-line" data-testid="tmdb-attribution">
        <img
          src={tmdbLogo}
          alt="TMDB"
          className="tmdb-logo"
          data-testid="tmdb-logo"
        />{" "}
        {TMDB_NOTICE}
      </p>
      <p className="attribution-line" data-testid="wikipedia-attribution">
        Plot text and remake evidence from Wikipedia, licensed{" "}
        <a
          href="https://creativecommons.org/licenses/by-sa/4.0/"
          target="_blank"
          rel="noreferrer"
        >
          CC BY-SA 4.0
        </a>
        ; every claim links its exact source revision.
      </p>
      <p className="attribution-line" data-testid="imdb-attribution">
        {IMDB_COURTESY} IMDb data is used under its non-commercial dataset
        terms — this is a non-commercial portfolio demo.
      </p>
    </footer>
  );
}
