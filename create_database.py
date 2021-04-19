import settings
import requests
import os
import json
import geocoder
import csv

dataset_field_names = ['tweet_id', 'username', 'location', 'text']
location_list = settings.countries_list
data_instances = list()
counter = dict()
ram_counter = 0


# Create dictionary for counting
def create_counter(thresh_total_instances):
    global counter
    # Instantiate counter
    for location in location_list:
        counter[location] = 0

    with open(settings.dataset_location, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            if line[2] in counter.keys():
                counter[line[2]] += 1
    print("Current database size: ", counter)

    # Remove country from location list if it exceeded threshold
    for location in location_list.copy():
        if counter[location] >= thresh_total_instances:
            location_list.remove(location)

# Creates headers to send
def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

# Gets rules whcih are currently stored on the server
def get_rules(headers):
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream/rules", headers=headers
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot get rules (HTTP {}): {}".format(response.status_code, response.text)
        )
    print("Downloaded server-stored rules")
    return response.json()

# Deletes rules which are currently stored on the server
def delete_all_rules(headers, rules):
    if rules is None or "data" not in rules:
        return None

    ids = list(map(lambda rule: rule["id"], rules["data"]))
    payload = {"delete": {"ids": ids}}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        headers=headers,
        json=payload
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot delete rules (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )
    print("Deleted server-stored rules")

# Sets new rules to the server
def set_rules(headers):
    # BOUNDING BOX FOR INDIA
    # TODO: probably delete
    india_loc = "7.710992,70.488281,30.297018,87.890625"

    # You can adjust the rules if needed
    sample_rules = [
        {"value": "mask"},
        {"value": "masks"}
    ]
    payload = {"add": sample_rules}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        headers=headers,
        json=payload,
    )
    if response.status_code != 201:
        raise Exception(
            "Cannot add rules (HTTP {}): {}".format(response.status_code, response.text)
        )
    print("Added new server-stored rules")


# Get tweet info
def get_tweet(json_response, headers):
    # Get tween by id
    response = requests.get(
        "https://api.twitter.com/1.1/statuses/lookup.json?tweet_mode=extended&id=" + json_response['data']['id'],
        headers=headers
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot get stream (HTTP {}): {}".format(response.status_code, response.text)
        )
    resp = response.json()
    # print(json.dumps(resp, indent=4, sort_keys=True))

    if resp == []:
        return False

    # Get info about tweet
    if 'retweeted_status' in resp[0].keys():
        full_text = resp[0]['retweeted_status']['full_text']
    else:
        full_text = resp[0]['full_text']
    return resp[0]['user']['screen_name'], resp[0]['user']['location'], full_text


def get_google_location(location, google_maps_api_key):
    try:
        result = geocoder.google(location, key=google_maps_api_key) # Catch errors
        return result.country_long
    except:
        return None


def save_in_csv(dataset_instances):
    global counter
    saved_ids = []

    if sum(list(counter.values())) == 0:
        with open(settings.dataset_location, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=dataset_field_names)
            writer.writeheader()

    with open(settings.dataset_location, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for i, line in enumerate(csv_reader):
            if i==0:
                continue
            saved_ids.append(line[0])

    with open(settings.dataset_location, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=dataset_field_names)

        for instance in dataset_instances:
            # CHECK FIRST IF INSTANCE IN DATASET, if it is, dont save it and continue:
            if instance[dataset_field_names[0]] in saved_ids:
                continue

            counter[instance[dataset_field_names[2]]] += 1
            try:
                writer.writerow(instance)
            except(exp):
                raise Exception(str(instance))


def save_tweet(tweet_id, username, location, text, saving_thresh=100, thresh_total_instances=2000):
    global counter, data_instances, ram_counter
    if counter[location] > thresh_total_instances:
        # Stop if the location has 2000 instances
        return

    # Add it to RAM
    data_instances.append({
        dataset_field_names[0]: tweet_id,
        dataset_field_names[1]: username,
        dataset_field_names[2]: location,
        dataset_field_names[3]: text
    })
    ram_counter += 1
    if ram_counter == saving_thresh:
        save_in_csv(data_instances)
        ram_counter = 0
        data_instances = list()


def get_stream(headers, google_maps_api_key, saving_thresh=100, thresh_totlal_instances=2000):

    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream", headers=headers, stream=True,
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot get stream (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )

    for response_line in response.iter_lines():
        if response_line:
            print("=" * 80)
            json_response = json.loads(response_line)
            tweet_id = json_response['data']['id']
            response = get_tweet(json_response, headers)
            if response == False:
                continue
            username, tweet_location, text = response
            location = get_google_location(tweet_location, google_maps_api_key)
            if location not in location_list:
                continue

            save_tweet(tweet_id, username, location, text, saving_thresh=saving_thresh, thresh_total_instances=thresh_totlal_instances)

            # print(json.dumps(json_response, indent=4, sort_keys=True))
            print(tweet_id)
            print(username)
            print(tweet_location, "=====>", location)
            print(text)

            counts = sum([counter[loc] for loc in location_list])
            if counts == thresh_totlal_instances*len(location_list):
                break



def main():
    bearer_token = settings.bearer_token
    google_maps_api_key = settings.google_maps_api_key
    saving_thresh = 5
    thresh_total_instances = 2000
    create_counter(thresh_total_instances=thresh_total_instances)

    headers = create_headers(bearer_token)
    rules = get_rules(headers)
    delete_all_rules(headers, rules)
    new_rules = set_rules(headers)
    get_stream(headers, google_maps_api_key, saving_thresh=saving_thresh, thresh_totlal_instances=thresh_total_instances)


if __name__ == "__main__":
    main()