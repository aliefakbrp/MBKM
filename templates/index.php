<!DOCTYPE html>
<html class="no-js" lang="en">
    <head>
        <!--- basic page needs
    ================================================== -->
        <meta charset="utf-8" />
        <title>{{ judulnya }}</title>
        <meta name="description" content="" />
        <meta name="author" content="" />

        <!-- mobile specific metas
    ================================================== -->
        <meta name="viewport" content="width=device-width, initial-scale=1" />

        <!-- CSS
    ================================================== -->
        <!-- <link rel="stylesheet" href= 'css/base.css'  />
        <link rel="stylesheet" href= 'css/vendor.css'  />
        <link rel="stylesheet" href= 'css/main.css'  /> -->
        <link rel="stylesheet" href="{{ url_for('static',filename='css/base.css') }}"   />
        <link rel="stylesheet" href="{{ url_for('static',filename='css/vendor.css') }}"   />
        <link rel="stylesheet" href="{{ url_for('static',filename='css/main.css') }}"   />

        <!-- script
    ================================================== -->
        <script src="{{ url_for('static',filename='js/modernizr.js')}}""></script>
        <script src="{{ url_for('static',filename='js/pace.min.js')}}""></script>

        <!-- favicons
    ================================================== -->
        <link rel="apple-touch-icon" sizes="180x180" href="apple-touch-icon.png" />
        <link rel="icon" type="image/png" sizes="32x32" href="favicon-32x32.png" />
        <link rel="icon" type="image/png" sizes="16x16" href="favicon-16x16.png" />
        <link rel="manifest" href="site.webmanifest" />
    </head>

    <body id="top">
        <div id="preloader">
            <div id="loader"></div>
        </div>

        <!-- site header
    ================================================== -->
        <header class="s-header">
            <div class="header-logo">
                <a class="site-logo" href="index.html">
                    <!-- <img src="{{ '../images/logo.svg' }}" alt="Homepage" /> -->
                    <!-- <img src="{{ url_for('static',filename='images/logo.svg') }}" alt="Homepage" /> -->
                    MBKM Riset 01
                </a>
            </div>

            <nav class="header-nav-wrap">
                <ul class="header-main-nav">
                    <li class="current"><a class="smoothscroll" href="#intro" title="intro">Intro</a></li>
                    <!-- <li><a class="smoothscroll" href="#about" title="about">About</a></li> -->
                    <li><a class="smoothscroll" href="#rekomendasi" title="services">Rekomendasi Film</a></li>
                    <!-- <li><a class="smoothscroll" href="#works" title="works">Works</a></li> -->
                    <!-- <li><a class="smoothscroll" href="#contact" title="contact us">Say Hello</a></li> -->
                </ul>

                <ul class="header-social">
                    <li>
                        <a href="#0"><i class="fab fa-facebook-f" aria-hidden="true"></i></a>
                    </li>
                    <li>
                        <a href="#0"><i class="fab fa-twitter" aria-hidden="true"></i></a>
                    </li>
                    <li>
                        <a href="#0"><i class="fab fa-dribbble" aria-hidden="true"></i></a>
                    </li>
                    <li>
                        <a href="#0"><i class="fab fa-behance" aria-hidden="true"></i></a>
                    </li>
                </ul>
            </nav>

            <a class="header-menu-toggle" href="#"><span>Menu</span></a>
        </header>
        <!-- end s-header -->

        <!-- intro
    ================================================== -->
        <section id="intro" class="s-intro target-section">
            <div class="row intro-content">
                <div class="column large-9 mob-full intro-text">
                    <h3>Hello, User 1</h3>
                    <h1>
                        Sistem Rekomendasi <br />
                        Film <br />
                        <!-- Based In Somewhere. -->
                    </h1>
                </div>

                <div class="intro-scroll">
                    <a href="#about" class="intro-scroll-link smoothscroll"> Scroll For More </a>
                </div>

                <div class="intro-grid"></div>
                <div class="intro-pic"></div>
            </div>
            <!-- end row -->
        </section>
        <!-- end intro -->

        <!-- about
    ================================================== -->
        <!-- end s-about -->

        <!-- services
            Rekomendasi
    ================================================== -->
        <section id="rekomendasi" class="s-services ss-dark target-section">
            <div class="shadow-overlay"></div>

            <div class="row heading-block heading-block--center" data-aos="fade-up">
                <div class="column large-full">
                    <h2 class="section-heading section-heading--centerbottom">Rekomendasi Film</h2>

                    <p class="section-desc">10 Rekomendasi Film untuk Kamu Tonton</p>
                </div>
            </div>
            <!-- end heading-block -->

            <div class="row services-list block-large-1-5 block-medium-1-2 block-tab-full">
                <div class="column item-service" data-aos="fade-up">
                    <div class="item-service__content">
                        <h4 class="item-title">Toy Story</h4>
                        <p>
                            keterangan film.
                        </p>
                    </div>
                </div>

                <div class="column item-service" data-aos="fade-up">
                    <div class="item-service__content">
                        <h4 class="item-title">Toy Story2</h4>
                        <p>
                            keterangan film.
                        </p>
                    </div>
                </div>

                <div class="column item-service" data-aos="fade-up">
                    <div class="item-service__content">
                        <h4 class="item-title">Toy Story 3</h4>
                        <p>
                            keterangan film.
                        </p>
                    </div>
                </div>

                <div class="column item-service" data-aos="fade-up">
                    <div class="item-service__content">
                        <h4 class="item-title">the Nun</h4>
                        <p>
                            keterangan film.
                        </p>
                    </div>
                </div>

                <div class="column item-service" data-aos="fade-up">
                    <div class="item-service__content">
                        <h4 class="item-title">The nun 2</h4>
                        <p>
                            keterangan film.
                        </p>
                    </div>
                </div>

                <div class="column item-service" data-aos="fade-up">
                    <div class="item-service__content">
                        <h4 class="item-title">The counjuring</h4>
                        <p>
                            keterangan film.
                        </p>
                    </div>
                </div>
                <div class="column item-service" data-aos="fade-up">
                    <div class="item-service__content">
                        <h4 class="item-title">The counjuring 2</h4>
                        <p>
                            keterangan film.
                        </p>
                    </div>
                </div>
                <div class="column item-service" data-aos="fade-up">
                    <div class="item-service__content">
                        <h4 class="item-title">The counjuring 3</h4>
                        <p>
                            keterangan film.
                        </p>
                    </div>
                </div>
                <div class="column item-service" data-aos="fade-up">
                    <div class="item-service__content">
                        <h4 class="item-title">The raid</h4>
                        <p>
                            keterangan film.
                        </p>
                    </div>
                </div>
                <div class="column item-service" data-aos="fade-up">
                    <div class="item-service__content">
                        <h4 class="item-title">The raid 2</h4>
                        <p>
                            keterangan film.
                        </p>
                    </div>
                </div>
            </div>
            <!-- end services-list -->
        </section>
        <!-- end s-services -->

        <!-- CTA
    ================================================== -->
        <section class="s-cta ss-dark">
            <div class="row heading-block heading-block--center" data-aos="fade-up">
                <div class="column large-full">
                    <h2 class="section-desc">Selamat Menonton Film yang kami rekomendasikan</h2>
                </div>
            </div>
            <!-- end heading-block -->

            <!-- <div class="row cta-content" data-aos="fade-up">
                <div class="column large-full">
                    <p>We highly recommend <a href="https://www.dreamhost.com/r.cgi?287326">DreamHost</a>. Powerful web and Wordpress hosting. Guaranteed. Starting at $2.95 per month.</p>

                    <a href="https://www.dreamhost.com/r.cgi?287326" class="btn full-width">Get Started</a>
                </div>
            </div> -->
            <!-- end ad-content -->
        </section>
        <!-- end s-cta -->


        <!-- footer
    ================================================== -->
        <footer>
            <div class="row">
                <div class="column large-full ss-copyright">
                    <span>© Copyright Epitome 2019</span>
                    <span>Design by <a href="https://www.styleshout.com/">StyleShout</a></span>
                </div>

                <div class="ss-go-top">
                    <a class="smoothscroll" title="Back to Top" href="#top"></a>
                </div>
            </div>
        </footer>

        <!-- photoswipe background
    ================================================== -->
        <div aria-hidden="true" class="pswp" role="dialog" tabindex="-1">
            <div class="pswp__bg"></div>
            <div class="pswp__scroll-wrap">
                <div class="pswp__container">
                    <div class="pswp__item"></div>
                    <div class="pswp__item"></div>
                    <div class="pswp__item"></div>
                </div>

                <div class="pswp__ui pswp__ui--hidden">
                    <div class="pswp__top-bar">
                        <div class="pswp__counter"></div>
                        <button class="pswp__button pswp__button--close" title="Close (Esc)"></button> <button class="pswp__button pswp__button--share" title="Share"></button>
                        <button class="pswp__button pswp__button--fs" title="Toggle fullscreen"></button> <button class="pswp__button pswp__button--zoom" title="Zoom in/out"></button>
                        <div class="pswp__preloader">
                            <div class="pswp__preloader__icn">
                                <div class="pswp__preloader__cut">
                                    <div class="pswp__preloader__donut"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="pswp__share-modal pswp__share-modal--hidden pswp__single-tap">
                        <div class="pswp__share-tooltip"></div>
                    </div>
                    <button class="pswp__button pswp__button--arrow--left" title="Previous (arrow left)"></button> <button class="pswp__button pswp__button--arrow--right" title="Next (arrow right)"></button>
                    <div class="pswp__caption">
                        <div class="pswp__caption__center"></div>
                    </div>
                </div>
            </div>
        </div>
        <!-- end photoSwipe background -->

        <!-- Java Script
    ================================================== -->
        <script src="{{ url_for('static',filename='js/jquery-3.2.1.min.js')}}""></script>
        <script src="{{ url_for('static',filename='js/plugins.js')}}""></script>
        <script src="{{ url_for('static',filename='js/main.js')}}""></script>
    </body>
</html>
