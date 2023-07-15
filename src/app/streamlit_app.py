############ 1. IMPORTING LIBRARIES ############

# Import streamlit, requests for API calls, and pandas and numpy for data manipulation
import re
import pickle
import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import predict
from streamlit_tags import st_tags  # to add labels on the fly!



############ 2. SETTING UP THE PAGE LAYOUT AND TITLE ############

# `st.set_page_config` is used to display the default layout width, the title of the app, and the emoticon in the browser tab.

st.set_page_config(
    layout="centered", page_title="Vietnamese Ecommerce product sentiment analysis", page_icon="❄️"
)

############ CREATE THE LOGO AND HEADING ############

# We create a set of columns to display the logo and the heading next to each other.


c1, c2 = st.columns([0.32, 2])

# The heading will be on the right.

with c2:

    st.caption("")
    st.title("Vietnamese Shopee product sentiment analysis")


# We need to set up session state via st.session_state so that app interactions don't reset the app.

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False

############ TABBED NAVIGATION ############

# First, we're going to create a tabbed navigation for the app via st.tabs()
# tabInfo displays info about the app.
# tabMain displays the main app.

MainTab, InfoTab = st.tabs(["Main", "Info"])

with InfoTab:

    st.subheader("What is this application about?")
    st.markdown(
        "This application is used for extract comments from shopee product and identify if that product is worth buying through user's comment "
    )

with MainTab:

    # Then, we create a intro text for the app, which we wrap in a st.markdown() widget.

    st.write("")
    st.markdown(
        """

    You do not have to read all the user's reviews to make your decision anymore!

    """
    )

    st.write("")

    # Now, we create a form via `st.form` to collect the user inputs.

    # All widget values will be sent to Streamlit in batch.
    # It makes the app faster!

    with st.form(key="my_form"):

        # The block of code below displays a text area
        # So users can paste their phrases to classify

        url = st.text_area(
            # Instructions
            "Enter shopee product url",
            # 'sample' variable that contains our keyphrases.
            "https://shopee.vn/-M%C3%A3-ELAS1015-gi%E1%BA%A3m-10-t%E1%BB%91i-%C4%91a-3TR-Apple-iPhone-13-128GB-i.288286284.12619089637",
            # The height
            height=100,
            # The tooltip displayed when the user hovers over the text area.
            help="Input the full url of the product from shopee",
            key="1",
        )

        # The block of code below:

        # 1. Converts the data st.text_area into a Python list.
        # 2. It also removes duplicates and empty lines.
        # 3. Raises an error if the user has entered more lines than in MAX_KEY_PHRASES.

        submit_button = st.form_submit_button(label="Submit")

    ############ CONDITIONAL STATEMENTS ############

    # Now, let us add conditional statements to check if users have entered valid inputs.
    # E.g. If the user has pressed the 'submit button without text, without labels, and with only one label etc.
    # The app will display a warning message.

    if not submit_button and not st.session_state.valid_inputs_received:
        st.stop()

    elif submit_button and not url:
        st.warning("Please input the product url")
        st.session_state.valid_inputs_received = False
        st.stop()

    elif submit_button or st.session_state.valid_inputs_received:

        if submit_button:

            # The block of code below if for our session state.
            # This is used to store the user's inputs so that they can be used later in the app.

            st.session_state.valid_inputs_received = True

        ############ MAKING THE API CALL ############

        # First, we create a Python function to construct the API call.

        def get_ratings(product_url):
            r = re.search(r"i\.(\d+)\.(\d+)", product_url)
            shop_id, item_id = r[1], r[2]
            ratings_url = "https://shopee.vn/api/v2/item/get_ratings?filter=1&flag=1&itemid={item_id}&limit=20&offset={offset}&shopid={shop_id}&type=0"
            offset = 0
            d = {"username": [], "rating": [], "comment": []}
            i = 1;
            while True:
                data = requests.get(
                    ratings_url.format(shop_id=shop_id, item_id=item_id, offset=offset)
                ).json()
                rating_comments_count = data['data']['item_rating_summary']['rcount_with_context']
                max_offset = rating_comments_count / 20;

                if i >= max_offset or data["data"]["ratings"] is None:
                    break

                for idx, rating in enumerate(data["data"]["ratings"]):
                    d["username"].append(rating["author_username"])
                    d["rating"].append(rating["rating_star"])
                    d["comment"].append(rating["comment"])

                offset += 20
                i = i + 1

            return pd.DataFrame(d)

        df_out = get_ratings(url)

        st.success("✅ Data extracted!")

        st.caption("")
        st.markdown("### Analysing the result!")
        st.caption("")

        # st.write(df)
        df_out['pred_sentiment'] = predict(df_out['comment']);

        # Display the dataframe
        st.write(df_out)

        cs, c1 = st.columns([2, 2])

        def plot_pie(data):
            data_label = data['pred_sentiment']
            data_label = pd.DataFrame(data_label, columns=['pred_sentiment']).groupby('pred_sentiment').size()
            fig, ax = plt.subplots()
            ax.pie(data_label, autopct="%.2f%%", labels=data_label.index)
            st.pyplot(fig)


        plot_pie(df_out)

        # The code below is for the download button
        # Cache the conversion to prevent computation on every rerun

        with cs:

            @st.experimental_memo
            def convert_df(df):
                return df_out.to_csv().encode("utf-8")

            csv = convert_df(df_out)

            st.caption("")

            st.download_button(
                label="Download results",
                data=csv,
                file_name="classification_results.csv",
                mime="text/csv",
            )